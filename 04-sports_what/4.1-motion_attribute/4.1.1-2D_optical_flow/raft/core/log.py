from visualdl import LogWriter


class Logger:
    # 思路 先push 等write
    def __init__(self, model, scheduler):
        self.writer = LogWriter(logdir="/home/aistudio/work/log/")
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.SUM_FREQ = SUM_FREQ
    
    def _print_training_status(self):
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ""
        for k in self.running_loss.keys():
            metrics_str += " %s: %f    " % (k, self.running_loss[k] / self.SUM_FREQ)
            self.writer.add_scalar(tag=k, step=int(self.total_steps % self.SUM_FREQ), value=self.running_loss[k] / self.SUM_FREQ)
        print(training_str, metrics_str)
        
    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        
        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def close(self):
        self.writer.close()