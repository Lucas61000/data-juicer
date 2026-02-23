#!/usr/bin/env python3
"""Generate synthetic test data for optimizer benchmarks."""

import argparse
import json
import random
import string
from pathlib import Path


def generate_test_data(output_path: str, num_samples: int = 1000, seed: int = 42):
    """Generate synthetic test data for benchmarking.

    Creates text samples with varying characteristics to test filters:
    - Various text lengths (some will be filtered by text_length_filter)
    - Various word counts (some will be filtered by words_num_filter)
    - Some with special characters (will be affected by special_characters_filter)
    - Some with repeated characters (will be affected by character_repetition_filter)
    """
    random.seed(seed)

    samples = []
    for i in range(num_samples):
        # Vary text length: 10-3000 chars (some will be filtered)
        text_len = random.randint(10, 3000)

        # Generate base text
        words = []
        current_len = 0
        while current_len < text_len:
            word_len = random.randint(2, 12)
            word = "".join(random.choices(string.ascii_lowercase, k=word_len))
            words.append(word)
            current_len += word_len + 1

        text = " ".join(words)[:text_len]

        # Add some structure (30% of samples)
        if random.random() < 0.3:
            text = f"Title: Sample {i}\n\n{text}\n\nConclusion: End of sample."

        # Add special characters to some samples (20%)
        if random.random() < 0.2:
            special_chars = "!@#$%^&*()[]{}|;:',.<>?/"
            insert_pos = random.randint(0, len(text) - 1)
            num_special = random.randint(5, 20)
            special = "".join(random.choices(special_chars, k=num_special))
            text = text[:insert_pos] + special + text[insert_pos:]

        # Add repeated characters to some samples (15%)
        if random.random() < 0.15:
            repeat_char = random.choice(string.ascii_lowercase)
            repeat_len = random.randint(15, 30)
            insert_pos = random.randint(0, len(text) - 1)
            text = text[:insert_pos] + (repeat_char * repeat_len) + text[insert_pos:]

        sample = {
            "text": text,
            "meta": {
                "id": i,
                "source": random.choice(["wiki", "arxiv", "web", "book"]),
            },
        }
        samples.append(json.dumps(sample, ensure_ascii=False))

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))

    print(f"Generated {num_samples} samples to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate test data for optimizer benchmarks")
    parser.add_argument("--output", "-o", type=str, default="test_data/benchmark_data.jsonl", help="Output file path")
    parser.add_argument("--samples", "-n", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    generate_test_data(args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
