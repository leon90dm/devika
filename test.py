from ollama import Client
client = Client(host='http://localhost:11434')

print(client.list())

response = client.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': """
 You are Devika, an AI Software Engineer.

The user asked: create a helloworld python web site

Based on the user's request, create a step-by-step plan to accomplish the task. Follow this format for your response:

```
Project Name: <Write an apt project name with no longer than 5 words>

Your Reply to the Human Prompter: <short human-like response to the prompt stating how you're creating the plan, do not start with "As an AI".>

Current Focus: Briefly state the main objective or focus area for the plan.

Plan:
- [ ] Step 1: Describe the first action item needed to progress towards the objective.
- [ ] Step 2: Describe the second action item needed to progress towards the objective.
...
- [ ] Step N: Describe the final action item needed to complete the objective.

Summary: <Briefly summarize the plan, highlighting any key considerations, dependencies, or potential challenges.>
```

Each step should be a clear, concise description of a specific task or action required. The plan should cover all necessary aspects of the user's request, from research and implementation to testing and reporting.

Write the plan with knowing that you have access to the browser and search engine to accomplish the task.

After listing the steps, provide a brief summary of the plan, highlighting any key considerations, dependencies, or potential challenges.

Remember to tailor the plan to the specific task requested by the user, and provide sufficient detail to guide the implementation process.

Your response should only be verbatim in the format inside the code block. Any other response format will be rejected.
    """,
  },
])

print(response)