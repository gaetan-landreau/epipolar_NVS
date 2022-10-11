import wandb


class logsTrain:
    def __init__(self, runName, idRun, config, resumeTraining=False):

        self.runName = runName
        self.idRun = idRun if idRun is not None else logsTrain.generateId()
        self.config = config
        self.resume = resumeTraining
        self.project = "NVSynthesis"
        self.initRun()

    @staticmethod
    def generateId():
        iDwandb = wandb.util.generate_id()
        return iDwandb

    def initRun(self):
        self.run = wandb.init(
            project="NVSynthesis",
            entity="author",
            group="epipolarNVS",
            id=self.idRun,
            config=self.config,
            resume=self.resume,
        )

        self.run.name = self.runName
        self.run.summary["RunId"] = self.idRun
        self.run.save()

    def getRun(self):
        return self.run


if __name__ == "__main__":
    log = logsTrain("test", None, {"test": "test"}, False)
