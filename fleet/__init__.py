"""Orchestration for running the PIAA finetune sweep across Lightning Studios.

Kept out of `src/`, which is the research code. Nothing here is imported by
training; the `lightning-sdk` dependency lives in its own `fleet` group so the
GPU machines never install it.
"""
