import random

random.seed(42)

from labels import LABELS

COLORS = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in LABELS}
