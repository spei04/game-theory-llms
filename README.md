# Studying LLM Cooperation Over Generations Through Iterated Donor Game Simulations

This repository contains a framework for simulating and analyzing iterated donor game interactions using Large Language Models (LLMs). The framework supports multiple game mechanics, strategy testing, and analysis capabilities. Code and game mechanices adapted and inspired by Aron Vallinder and Edward Hughes: https://arxiv.org/abs/2412.10270

## Overview

The framework is a multi-generation simulation where:
- Agents are given an initial endowment of resources (typically 10 units)
- In each round, agents are randomly paired as donor/recipient
- Donors decide how much to donate to their recipient
- Recipients receive 2x the donated amount
- Agents alternate between donor/recipient roles
- Top-performing agents survive to the next generation

## Repository Structure

The repository is organized into three main directories:

### **simulations/**
Start here! Contains core simulation and testing scripts for the iterated donor game experiments. Includes baseline implementations, enhanced versions with history settings, fixed endowment scenarios, and strategy testing files.

*See [simulations/README.md](simulations/README.md) for detailed file descriptions.*

### **figures/**
Contains visualization and analysis scripts for processing simulation results and generating plots.

*See [figures/README.md](figures/README.md) for detailed file descriptions.*

### **modular system/**
Contains the advanced modular framework for testing LLM models in different roles within the simulation.

*See [modular system/README.md](modular%20system/README.md) for detailed file descriptions.*

## Setup

1. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**
   
   Create a `.env` file in the root directory:
   ```bash
   # OpenAI API
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Anthropic API  
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Google Generative AI API
   GOOGLE_API_KEY=your_google_api_key_here
   ```
