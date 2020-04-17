
class BaseLogger(object):
    """
    Base logger handler. See implementations: TensorboardLogger, VisdomLogger, PolyaxonLogger, MLflowLogger, ...

    """
    def log_metrics(self, metrics, step):

        """Record metrics.
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError()

    def log_aggregate(self, agg, step):

        """Record metrics.
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
        """
        raise NotImplementedError()



    def save(self):

        """Save log data."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass





        pass

