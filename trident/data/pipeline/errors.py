from __future__ import absolute_import, division, print_function


class DataPipelineError(RuntimeError):
    """Structured error raised with the location of a failed pipeline stage."""

    error_code = "data_pipeline_error"

    def __init__(self, message, stage=None, sample_id=None, field=None, cause=None):
        self.raw_message = message
        self.stage = stage
        self.sample_id = sample_id
        self.field = field
        self.cause = cause
        details = []
        if stage is not None:
            details.append("stage={0}".format(stage))
        if sample_id is not None:
            details.append("sample_id={0}".format(sample_id))
        if field is not None:
            details.append("field={0}".format(field))
        if cause is not None:
            details.append("cause={0}".format(cause))
        suffix = " ({0})".format(", ".join(details)) if details else ""
        RuntimeError.__init__(self, "{0}{1}".format(message, suffix))
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self):
        return dict(
            code=self.error_code, message=self.raw_message, stage=self.stage,
            sample_id=self.sample_id, field=self.field,
            cause=None if self.cause is None else str(self.cause),
        )


class SchemaValidationError(DataPipelineError):
    error_code = "schema_validation_error"

    def __init__(self, message, field=None, sample_id=None, cause=None):
        DataPipelineError.__init__(
            self, message, stage="schema", field=field,
            sample_id=sample_id, cause=cause)


class SourceError(DataPipelineError):
    error_code = "source_error"


class TransformError(DataPipelineError):
    error_code = "transform_error"


class CollationError(DataPipelineError):
    error_code = "collation_error"


class MemoryBudgetError(DataPipelineError):
    error_code = "memory_budget_error"