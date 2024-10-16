"""A custom SequenceClassificationModel for multi-task classification."""

from typing import Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiTaskConfig(PretrainedConfig):
    """A custom configuration for multi-task classification of disease and amyloid status."""

    model_type = "multi_task"

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the configuration.

        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)


class MultiTaskSequenceClassificationModel(PreTrainedModel):
    """A custom SequenceClassificationModel for multi-task classification of disease and amyloid status."""

    config_class = MultiTaskConfig

    def __init__(self, model_name, config, cache_dir=None):
        """Initialize the model.

        :param model_name: The name of the base model.
        :type model_name: str
        :param config: The configuration.
        :type config: PretrainedConfig
        :param cache_dir: The cache directory.
        :type cache_dir: Optional[str]
        """
        # Pass the config to the parent class
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        # Initialize the original model using the config
        self.model = AutoModel.from_pretrained(
            model_name,
            config,
            cache_dir=cache_dir,
        )
        # Use two classification heads for the two combined classification tasks
        # Note that the model itself might not be a Roberta model but
        # the head should be sufficiently general
        self.classifier_task1 = RobertaClassificationHead(config)
        self.classifier_task2 = RobertaClassificationHead(config)

        # Add the classifiers to named modules
        self.add_module("classifier_task1", self.classifier_task1)
        self.add_module("classifier_task2", self.classifier_task2)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_ids: torch.LongTensor = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """Forward pass of the model.

        :param input_ids: The input IDs.
        :type input_ids: Optional[torch.LongTensor]
        :param attention_mask: The attention mask.
        :type attention_mask: Optional[torch.FloatTensor]
        :param token_type_ids: The token type IDs.
        :type token_type_ids: Optional[torch.LongTensor]
        :param position_ids: The position IDs.
        :type position_ids: Optional[torch.LongTensor]
        :param head_mask: The head mask.
        :type head_mask: Optional[torch.FloatTensor]
        :param inputs_embeds: The input embeddings.
        :type inputs_embeds: Optional[torch.FloatTensor]
        :param labels: The labels.
        :type labels: Optional[torch.LongTensor]
        :param output_attentions: Whether to output attentions.
        :type output_attentions: Optional[bool]
        :param output_hidden_states: Whether to output hidden states.
        :type output_hidden_states: Optional[bool]
        :param return_dict: Whether to return a dictionary.
        :type return_dict: Optional[bool]
        :param task_ids: The task IDs.
        :type task_ids: torch.LongTensor
        :return: The logits or the output.
        :rtype: Union[Tuple[torch.Tensor], SequenceClassifierOutput]
        :raises ValueError: If the task is not supported.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # The idea is that the batches are mostly kept together by task, but sometimes they are mixed
        # In those cases we need to pass the correct parts of the sequence_output to the correct heads
        logits = []
        unique_task_ids = []
        # Get the tasks in the same order as they appear in the batch for the correct order
        # for the concatenation of the logits
        for task_id in task_ids:
            if task_id not in unique_task_ids:
                unique_task_ids.append(int(task_id))

        # Loop over the unique task IDs and get the logits for each task using the corresponding head for the task
        for task_id in unique_task_ids:
            if task_id == 0:
                logits.append(self.classifier_task1(sequence_output[task_ids == task_id]))
            elif task_id == 1:
                logits.append(self.classifier_task2(sequence_output[task_ids == task_id]))
            else:
                raise ValueError(f"Task ID {task_id} is not supported.")

        # Concatenate the logits
        logits = torch.cat(logits, dim=0)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)  # type: ignore
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # type: ignore
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # type: ignore
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
