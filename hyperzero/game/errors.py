"""Game engine exceptions."""


class GameError(Exception):
    """Base class for game engine errors."""


class InvalidActionError(GameError):
    """Raised when an action is outside the action space or currently illegal."""


class TerminalStateError(GameError):
    """Raised when attempting to advance a terminal state."""
