import click

import app.tasks  # noqa: F401
from app.internal.settings import Settings
from app.internal.taskhandler import TaskHandlerFactory


@click.group()
def cli():
    """
    CLI interface for handling timeseries model tasks.
    """
    pass


@click.command("run")
@click.argument(
    "json_config",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
    ),
)
def run(json_config):
    """
    Runs a set of tasks specified in the configuration file.
    """
    settings = Settings.read(json_config)

    handlers = [
        TaskHandlerFactory().factory(
            t["kind"],
            definition,
            settings.dataset,
            settings.sampling,
            settings.get_model_definition(),
        )
        for t, definition in zip(
            settings.tasks, settings.get_tasks_definitions()
        )
    ]

    for handler in handlers:
        handler.preprocess()
        handler.run()
        handler.postprocess()


cli.add_command(run)
