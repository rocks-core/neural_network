import wandb


class WandbLogger:
	def __init__(self, metrics, project_name="neural_network"):
		self.metrics = metrics
		wandb.init(project=project_name)

	def __call__(self, model, *args, **kwargs):
		# fold = ""
		# if "fold" in kwargs:
		# 	fold = "Fold_" + kwargs["fold"] + "_"
		if self.metrics == "all":
			selected_metrics = model.metrics_score
		else:
			selected_metrics = {}
			for m, v in model.metrics_score.items():
				if m in self.metrics:
					selected_metrics[m] = v
		wandb.log(selected_metrics)
