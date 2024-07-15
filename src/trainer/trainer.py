import numpy

from src.trainer.thunder_trainer import ThunderTrainer


class Trainer(ThunderTrainer):
    def _forward(self, x):
        # print(x.shape)
        x_hat = self.model(x)
        return x_hat

    def _training_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(y.shape[0], -1)

        if self.distributed:
            # send batch to the GPU
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

        self.optimizer.zero_grad()
        x_hat = self._forward(x)
        loss = self.criterion(x_hat, y)
        loss.backward()
        self.optimizer.step()

        return loss

    def _validation_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(y.shape[0], -1)

        if self.distributed:
            # send batch to the GPU
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

        x_hat = self._forward(x)
        loss = self.criterion(x_hat, y)

        return loss

    def _test_step(self, batch, batch_index):
        x, y = batch
        y = y.reshape(y.shape[0], -1)

        if self.distributed:
            # send batch to the GPU
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

        x_hat = self._forward(x)
        loss = self.criterion(x_hat, y)

        # if master log predictions
        if self.is_master:
            self.logger.log_csv(
                "predictions",
                ["real", "predictions"],
                [y.detach().cpu().squeeze().tolist(), x_hat.detach().cpu().squeeze().tolist()],
                self.current_epoch
            )

        return loss
