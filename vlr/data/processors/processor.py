class Processor:
    """
    This class and its children are used to perform specific tasks to process data.
    """
    def process_sample(self, sample: dict, *args, **kwargs) -> dict:
        """
        Process sample.
        :param sample:      Sample.
        :param args:        Additional arguments.
        :param kwargs:      Additional keyword arguments.
        :return:            Processed sample.
        """
        raise NotImplementedError("Please implement this method in child classes")

    def process_batch(self, batch: dict, *args, **kwargs) -> dict:
        """
        Process batch of data.
        :param batch:       Batch of data.
        :param args:        Additional arguments.
        :param kwargs:      Additional keyword arguments.
        :return:            Processed batch.
        """
        raise NotImplementedError("Please implement this method in child classes")

    def check_output(self):
        """
        Check output.
        """
        raise NotImplementedError("Please implement this method in child classes")
