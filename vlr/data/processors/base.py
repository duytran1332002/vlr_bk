class Processor:
    """
    This class and its children are used to perform specific tasks to process data.
    """
    def process_sample(self, sample: dict):
        """
        Process sample.
        :param sample:  Sample.
        :return:        Processed sample.
        """
        raise NotImplementedError("Please implement this method in child classes")

    def process_batch(self):
        """
        Process batch of data.
        """
        raise NotImplementedError("Please implement this method in child classes")
