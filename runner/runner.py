import os
import sys
sys.path.append("../")
import logging
logger = logging.getLogger(__name__)

from metric import eval_std_f1_score, eval_text_f1_score, eval_head_f1_score, show_results
from runner.train import Trainer
from runner.evaluate import Evaluator


class BaseRunner:
    def __init__(
        self,
        cfg=None,
        data_samples=None,
        data_features=None,
        data_loaders=None,
        model=None,
        optimizer=None,
        scheduler=None,
        metric_fn_dict=None,
    ):

        self.cfg = cfg
        self.model = model
        self.train_samples, self.dev_samples, self.test_samples = data_samples
        self.train_features, self.dev_features, self.test_features = data_features
        self.train_loader, self.dev_loader, self.test_loader = data_loaders

        self.trainer = Trainer(
            cfg=self.cfg,
            data_loader=self.train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.dev_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=self.dev_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=self.dev_features,
            set_type = "DEV",
            invalid_num=self.cfg.dev_invalid_num,
        )
        self.test_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=self.test_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=self.test_features,
            set_type = "TEST",
            invalid_num=self.cfg.test_invalid_num,
        )


    def run(self):
        if self.cfg.inference_only:
            self.inference()
        else:
            self.train()

    
    def train(self):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_loader)*self.cfg.batch_size)
        logger.info("  batch size = %d", self.cfg.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.cfg.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.cfg.max_steps)

        for global_step in range(self.cfg.max_steps):
            self.trainer.train_one_step()

            if (global_step+1)%self.cfg.logging_steps == 0:
                self.trainer.write_log()

            if (global_step+1)%self.cfg.eval_steps==0:
                self.eval_and_update(global_step)

    
    def inference(self):
        dev_c, _ = self.dev_evaluator.evaluate()
        test_c, _ = self.test_evaluator.evaluate()
        self.report_result(dev_c, test_c)


    def save_checkpoints(self):
        cpt_path = os.path.join(self.cfg.output_dir, 'checkpoint')
        if not os.path.exists(cpt_path):
            os.makedirs(cpt_path)
        self.model.save_pretrained(cpt_path)


    def eval_and_update(self, global_step):
        raise NotImplementedError()


    def report_result(self, dev_c, test_c, global_step=None):
        raise NotImplementedError()


class Runner(BaseRunner):
    def __init__(self, cfg=None, data_samples=None, data_features=None, data_loaders=None, model=None, optimizer=None, scheduler=None, metric_fn_dict=None):
        super().__init__(cfg, data_samples, data_features, data_loaders, model, optimizer, scheduler, metric_fn_dict)
        self.metric = {
            "best_dev_f1": 0.0,
            "related_test_f1": 0.0,
        }

        self.metric_fn_dict = {
            "span": eval_std_f1_score,
            "text": eval_text_f1_score,
            "head": eval_head_f1_score,
        }
        self.dev_evaluator.metric_fn_dict = self.metric_fn_dict
        self.test_evaluator.metric_fn_dict = self.metric_fn_dict


    def eval_and_update(self, global_step):
        dev_c, _ = self.dev_evaluator.evaluate()
        test_c, _ = self.test_evaluator.evaluate()

        output_dir = os.path.join(self.cfg.output_dir, 'checkpoint')
        os.makedirs(output_dir, exist_ok=True)

        dev_f1, test_f1 = dev_c["f1"], test_c["f1"]
        if dev_f1 > self.metric["best_dev_f1"]:
            self.metric["best_dev_f1"] = dev_f1
            self.metric["related_test_f1"] = test_f1

            self.report_result(dev_c, test_c, global_step)   
            self.save_checkpoints()
        logger.info('current best dev-f1 score: {}'.format(self.metric["best_dev_f1"]))
        logger.info('current related test-f1 score: {}'.format(self.metric["related_test_f1"]))


    def report_result(self, dev_c, test_c, global_step=None):
        show_results(self.test_features, os.path.join(self.cfg.output_dir, f'best_test_related_results.log'), 
            {"test related best score": f"P: {test_c['precision']} R: {test_c['recall']} f1: {test_c['f1']}", "global step": global_step}
        )
        show_results(self.dev_features, os.path.join(self.cfg.output_dir, f'best_dev_results.log'), 
            {"dev best score": f"P: {dev_c['precision']} R: {dev_c['recall']} f1: {dev_c['f1']}", "global step": global_step}
        )