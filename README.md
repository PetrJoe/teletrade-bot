# TeleTrade Bot

Trading bots with machine learning...

## Development setup

### Prerequisites

- This project uses [uv](https://docs.astral.sh/uv/) to manage its dependencies. If you don't
have it installed, install it from the [installation script](https://docs.astral.sh/uv/getting-started/installation/) 
or with `pip` globally `pip install uv`

- Create a `.env` file in  the project root and populate using the `.env.sample` as a guide  to the  required  
environmental variables for the smooth execution of the bots.

### Install dependencies

- Install the project's dependencies by running `uv sync`

### Activate you virtual environment

- After a successful installation of the project dependencies, a `.venv` directory is created in the project's
root activate it according to your operating system. (mac & linux `source .venv/bin/activate`)

### Run the bot scripts

Two entry points have been made available to run the bots. `bots` and `tbot`. 

#### Running `bots` bot
```bash
$ uv run bots
```
#### Running `tbot` bot
```bash
$ uv run tbot
```