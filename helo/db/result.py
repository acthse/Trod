class ExeResult:

    __slots__ = ('affected', 'last_id')

    def __init__(self, affected: int, last_id: int) -> None:
        self.affected = affected
        self.last_id = last_id

    def __repr__(self) -> str:
        return f"ExeResult(affected: {self.affected}, last_id: {self.last_id})"

    def __str__(self) -> str:
        return f"({self.affected}, {self.last_id})"
