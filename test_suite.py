import unittest
import torch
import os
import shutil

from netcradus_llm.config import NetcradusConfig, PROTOTYPE_CONFIG
from netcradus_llm.tokenizer import NetcradusTokenizer
from netcradus_llm.model import NetcradusForCausalLM, GroupedQueryAttention, RotaryEmbedding
from netcradus_llm.dataset import PretrainingDataset, SFTDataset
from netcradus_llm.train import NetcradusTrainer
from netcradus_llm.inference import NetcradusPipeline


class TestNetcradusLLM(unittest.TestCase):

    def setUp(self):
        self.config = NetcradusConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256
        )
        self.tokenizer = NetcradusTokenizer(vocab_size=self.config.vocab_size)

    def test_tokenizer_special_tokens(self):
        text = "<|im_start|>user\nHello World!<|im_end|>"
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(encoded[0], self.tokenizer.special_tokens["<|im_start|>"])
        self.assertEqual(encoded[-1], self.tokenizer.special_tokens["<|im_end|>"])

        decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
        self.assertIn("<|im_start|>", decoded)
        self.assertIn("<|im_end|>", decoded)

    def test_model_forward_and_loss(self):
        model = NetcradusForCausalLM(self.config)
        input_ids = torch.randint(8, self.config.vocab_size, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        self.assertIsNotNone(outputs["loss"])
        self.assertEqual(outputs["logits"].shape, (2, 16, self.config.vocab_size))

    def test_gqa_kv_cache(self):
        model = NetcradusForCausalLM(self.config)
        model.eval()

        prompt_ids = torch.randint(8, self.config.vocab_size, (1, 10))
        with torch.no_grad():
            out1 = model(prompt_ids, use_cache=True)
            past_kv = out1["past_key_values"]
            self.assertEqual(len(past_kv), self.config.num_hidden_layers)

            next_token = torch.randint(8, self.config.vocab_size, (1, 1))
            out2 = model(next_token, past_key_values=past_kv, use_cache=True)
            self.assertEqual(out2["logits"].shape, (1, 1, self.config.vocab_size))

    def test_generation(self):
        model = NetcradusForCausalLM(self.config)
        input_ids = torch.randint(8, self.config.vocab_size, (1, 5))
        generated = model.generate(input_ids, max_new_tokens=10, temperature=0.7, top_p=0.9, top_k=20)
        self.assertEqual(generated.shape[1], 15)

    def test_dataset_and_trainer(self):
        texts = ["Netcradus LLM is designed for fast and scalable AI training."] * 5
        dataset = PretrainingDataset(texts, self.tokenizer, max_seq_len=32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model = NetcradusForCausalLM(self.config)
        output_dir = "./tmp_checkpoint_test"
        trainer = NetcradusTrainer(
            model=model,
            train_dataloader=dataloader,
            max_steps=2,
            output_dir=output_dir
        )
        res = trainer.train()
        self.assertIn("final_loss", res)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def test_pipeline(self):
        model = NetcradusForCausalLM(self.config)
        pipeline = NetcradusPipeline(model, self.tokenizer)
        res = pipeline.generate("Hello world", max_new_tokens=10)
        self.assertIsInstance(res, str)


if __name__ == "__main__":
    unittest.main()
