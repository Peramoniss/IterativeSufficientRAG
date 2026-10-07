class IRCoTPromptTemplate:
    def __init__(self, user_prompt, system_prompt="", reference_template="", question_template="", reference_label=False):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.reference_template = reference_template
        self.question_template = question_template
        self.reference_label = reference_label

    def _format_reference(self, docs):
        reference = "Reference:\n" if self.reference_label  else ""
        for doc in docs:
            reference += self.reference_template.format(**doc)
            reference += "\n\n"
        return reference

    def format_prompt(self, question, retrieved_documents=[], prefix="", suffix=""):
        formatted_reference = self._format_reference(retrieved_documents)
        input_params = {
            "question": question,
            "reference": formatted_reference,
            "prefix": prefix,
            "suffix": suffix
        }
        user_prompt = self.user_prompt.format(**input_params)

        messages = [{"role": "user", "content": user_prompt}]
        if self.system_prompt and len(self.system_prompt) > 0:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

class PromptTemplate:
    def __init__(self, user_prompt, system_prompt=""):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def format_prompt(self, input_params):
        user_prompt = self.user_prompt.format(**input_params)

        messages = [{"role": "user", "content": user_prompt}]
        if self.system_prompt and len(self.system_prompt) > 0:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages