# Modified from https://github.com/ShangtongZhang/DeepRL

class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


class LinearSchedule:
    def __init__(self, start, end=None, steps=None, steps_per_iter=1):
        if end is None:
            end = start
            steps = 1
        if type(steps) is tuple:
            self.step_start = steps[0]
            n_steps = steps[1] - steps[0]
        else:
            self.step_start = 0
            n_steps = steps

        try:
            self.inc = (end - start) / float(n_steps)
        except ZeroDivisionError:
            self.inc = 0

        self.current = start
        self.end = end
        self.curr_step = 0
        if end > start:
            self.bound = min
        else:
            self.bound = max
        self.steps_per_iter = steps_per_iter  # steps per iter

    def __call__(self, steps=None):
        if steps is None:
            steps = self.steps_per_iter
        val = self.current
        self.curr_step += steps
        if self.curr_step >= self.step_start:
            self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
