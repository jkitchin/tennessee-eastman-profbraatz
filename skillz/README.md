# TEP Simulator Skills

This directory contains skill/instruction files for AI coding assistants to work effectively with the Tennessee Eastman Process (TEP) simulator.

## Contents

- `TEP/SKILL.md` - Main skill definition with quick start examples
- `TEP/simulator-reference.md` - TEPSimulator API reference
- `TEP/process-variables.md` - XMEAS and XMV variable reference
- `TEP/fault-detection.md` - Fault detection framework guide
- `TEP/cli-reference.md` - Command-line interface documentation

## Installation by Platform

### Claude Code

Copy or symlink the `TEP` directory to your Claude Code skills location:

```bash
# Create skills directory if it doesn't exist
mkdir -p ~/.claude/skills

# Symlink the TEP skill
ln -s /path/to/tep-python/skillz/TEP ~/.claude/skills/TEP
```

The skill will be available as `tep-simulator` in Claude Code sessions.

### OpenAI (ChatGPT / Custom GPTs)

For Custom GPTs, concatenate the skill files and add to the GPT's instructions:

```bash
# Concatenate all skill files
cat TEP/SKILL.md TEP/simulator-reference.md TEP/process-variables.md \
    TEP/fault-detection.md TEP/cli-reference.md > tep-skill-combined.md
```

Then paste the contents into your Custom GPT's "Instructions" field, or upload as a knowledge file.

### Google Gemini

For Gemini with Google AI Studio or API:

1. **System Instruction**: Copy the contents of `TEP/SKILL.md` into the system instruction
2. **Context Files**: Upload the reference files as context documents

```bash
# Or create a single combined file
cat TEP/*.md > tep-gemini-context.md
```

### OpenCode

For [OpenCode](https://github.com/opencode-ai/opencode), add the skill to your project's `.opencode/skills` directory:

```bash
# Create skills directory
mkdir -p .opencode/skills

# Copy the TEP skill
cp -r /path/to/tep-python/skillz/TEP .opencode/skills/
```

Or add a symlink in your global OpenCode configuration:

```bash
mkdir -p ~/.opencode/skills
ln -s /path/to/tep-python/skillz/TEP ~/.opencode/skills/TEP
```

### Cursor / Continue.dev

For Cursor or Continue.dev, add the skill content to your project's documentation context:

```bash
# Copy to project docs
mkdir -p docs/ai-context
cp -r /path/to/tep-python/skillz/TEP docs/ai-context/

# Or symlink
ln -s /path/to/tep-python/skillz/TEP docs/ai-context/TEP
```

Then configure the AI assistant to include `docs/ai-context/` in its context.

### Cline / Aider

For Cline or Aider, include the skill files in your prompt or context:

```bash
# Aider - add to repo map
aider --read TEP/SKILL.md --read TEP/simulator-reference.md

# Or add to .aider.conf.yml
read:
  - skillz/TEP/SKILL.md
  - skillz/TEP/simulator-reference.md
```

## Verifying Installation

After installation, test by asking your AI assistant:

> "How do I run a TEP simulation with fault IDV(4) starting at 2 hours?"

The assistant should respond with code using `TEPSimulator` and the `disturbances` parameter.

## Customization

Feel free to modify the skill files for your specific use case:

- Add project-specific examples
- Remove sections not relevant to your work
- Add custom fault detection or control strategies
