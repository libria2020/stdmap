class Metrics:
    def __init__(self, metrics: str):
        self.metric = metrics
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def __iadd__(self, other):
        self.value = other
        self.sum += other
        self.count += 1
        self.average = self.sum / self.count
        return self

    def __repr__(self):
        return str(self.average)

    def __format__(self, format_spec):
        return format(self.average, format_spec)

    @property
    def name(self):
        return self.metric

    @property
    def last_value(self):
        return self.value

    @property
    def score(self):
        return self.average
