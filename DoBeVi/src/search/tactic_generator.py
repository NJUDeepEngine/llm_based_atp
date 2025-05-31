from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import torch
import ray
import logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
import uuid
import asyncio

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

class ModelEmptyOutputError(Exception):
    """Raised when the model returns no output (empty result)."""
    pass

class TacticGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
    
@ray.remote(num_gpus=1)
class HuggingFaceGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.decoder_only = False
        except ValueError:
            self.generator = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            self.decoder_only = True
        self.generator.eval()

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        gen_kwargs = dict(
            max_length=self.max_length,
            num_beams=num_samples,
            num_return_sequences=num_samples,
            length_penalty=self.length_penalty,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.generator.generate(
            **inputs,
            **gen_kwargs,
        )

        outputs_score = outputs.sequences_scores.tolist()
        outputs = outputs.sequences.view(num_samples, -1)

        output_text = []
        output_score = []
        for i in range(num_samples):
            model_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            if self.decoder_only:
                model_output = model_output.replace(state, "").strip().strip("\n").strip()
            else:
                model_output = model_output.strip().strip("\n").strip()
            output_text.append(model_output)
            log_prob_score = outputs_score[i]
            output_score.append(log_prob_score)
        
        return list(zip(output_text, output_score))
    
    
    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            logging.warning(f"Input length exceeds max_length: {inputs['input_ids'].size(1)}")
            return []
        gen_kwargs = dict(
            max_length=self.max_length,
            num_beams=num_samples,
            num_return_sequences=num_samples,
            length_penalty=self.length_penalty,
            early_stopping=True,
            repetition_penalty=1.1,
            do_sample=True,
            temperature=1.1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.generator.generate(
            **inputs,
            **gen_kwargs,
        )

        outputs_score = outputs.sequences_scores.tolist()
        outputs = outputs.sequences.view(num_samples, -1)

        output_text = []
        output_score = []
        for i in range(num_samples):
            model_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            if self.decoder_only:
                model_output = model_output.replace(state, "").strip().strip("\n").strip()
            else:
                model_output = model_output.strip().strip("\n").strip()
            output_text.append(model_output)
            log_prob_score = outputs_score[i]
            output_score.append(log_prob_score)
        
        return list(zip(output_text, output_score))

@ray.remote(num_gpus=1)
class VllmGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        gen_params = SamplingParams(
            n=num_samples,
            temperature=0,
            repetition_penalty=1.1,
            logprobs=0,
        )
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions

    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        gen_params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=128,
            logprobs=0,
        )
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions
    
    async def _run_query(
        self,
        state: str
    ) -> Tuple[str, float]:
        gen_params = SamplingParams(
            n=1,
            temperature=0,
            top_k=50,
            top_p=0.95,
            max_tokens=128,
            logprobs=0,
        )
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        resp = output.outputs[0].text.strip('\n') # Can not use strip() here
        score = output.outputs[0].cumulative_logprob or 0.0
        return state + resp, score
    
    async def batch_generate_sampling(
        self,
        states: List[str],
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(states, return_tensors="pt", padding=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        tasks = [asyncio.create_task(self._run_query(state)) for state in states]
        suggestions = []
        for task in asyncio.as_completed(tasks):
            resp, score = await task
            suggestions.append((resp, score))
        return suggestions
    
@ray.remote(num_gpus=1)
class InternlmVllmGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        length_penalty: float,
        max_length: int,
        gpu_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.length_penalty = length_penalty
        self.max_length = max_length
        self.gpu_id = gpu_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # init vllm
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=self.max_length,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(state, return_tensors="pt", truncation=True).to("cuda")
        if inputs["input_ids"].size(1) > self.max_length:
            return []
        gen_params = SamplingParams(
            n=num_samples,
            temperature=0,
            repetition_penalty=1.1,
            logprobs=0,
        )
        async for output in self.engine.generate(state, gen_params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        suggestions = [(o.text.strip().strip("\n"), o.cumulative_logprob or 0.0) for o in output.outputs]
        return suggestions

    async def generate_sampling(
        self,
        state: str,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        
        def prompt_style_internlm_chat_stepprover_extractor(result:str):
            return result
    
        def _unique_sorted(texts, scores):
            texts_ = []
            scores_ = []
            for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
                if t not in texts_:
                    texts_.append(t)
                    scores_.append(s)
            return texts_, scores_
    
        texts, scores = [], []
        params = SamplingParams(
            n=num_samples,
            temperature=1.1,
            max_tokens=128,
            stop=['<|im_end|>',],
            logprobs=True,
        )
        
        async for output in self.engine.generate(state, params, request_id=str(uuid.uuid4().hex)):
            final_output = output
        output = final_output
        if not isinstance(output, RequestOutput):
            raise ValueError(f"Expected RequestOutput, got {type(output)}")
        
        for o in output.outputs:
            text = o.text.replace(self.tokenizer.eos_token, '').strip().strip("\n")
            score = o.cumulative_logprob or 0.0
            texts.append(text)
            scores.append(score)

        texts = list(map(prompt_style_internlm_chat_stepprover_extractor,texts))
        texts, scores = _unique_sorted(texts, scores)
        suggestions = []
        for i in range(len(texts)):
            suggestions.append((texts[i], scores[i]))
    
        return suggestions