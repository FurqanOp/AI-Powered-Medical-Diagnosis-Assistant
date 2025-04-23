import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
os.environ["GOOGLE_API_KEY"] = "AIzaSyAVRxwF-hdwPSa16pZWh0rnW80T_gOH9V4"


# Load environment variables
load_dotenv()

# Define the AI Model
llm = LLM(model="gemini/gemini-1.5-flash",
          verbose=True,
          temperature=0.5,
          api_key=os.environ["GOOGLE_API_KEY"])

# Read the medical report
with open("C:\Python Flask\Medical Rerort - Michael Johnson - Panic Attack Disorder.txt") as file:
    medical_report = file.read()

# Specialist Agents
cardiologist = Agent(
    role='Cardiologist',
    goal='Analyze the report for cardiovascular concerns.',
    backstory='An experienced cardiologist with expertise in heart-related conditions.',
    llm=llm,
    tools=[],
    allow_delegation=True
)

psychologist = Agent(
    role='Psychologist',
    goal='Assess the report for psychological concerns.',
    backstory='A seasoned psychologist specialized in panic disorders.',
    llm=llm,
    tools=[],
    allow_delegation=True
)

pulmonologist = Agent(
    role='Pulmonologist',
    goal='Identify respiratory concerns in the report.',
    backstory='A pulmonologist with a focus on lung and breathing disorders.',
    llm=llm,
    tools=[],
    allow_delegation=True
)

# Tasks
task_cardiologist = Task(
    description='Analyze cardiovascular issues.',
    agent=cardiologist,
    expected_output='Detailed cardiovascular report.'
)

task_psychologist = Task(
    description='Analyze psychological issues.',
    agent=psychologist,
    expected_output='Detailed psychological report.'
)

task_pulmonologist = Task(
    description='Analyze respiratory issues.',
    agent=pulmonologist,
    expected_output='Detailed respiratory report.'
)

# Multidisciplinary Team Agent
team_agent = Agent(
    role='Multidisciplinary Team',
    goal='Combine insights from all specialists to provide a final diagnosis.',
    backstory='An experienced medical coordinator who ensures accurate diagnosis.',
    llm=llm,
    tools=[],
    allow_delegation=True
)

task_team_diagnosis = Task(
    description='Combine all reports to generate a final diagnosis.',
    agent=team_agent,
    expected_output='Final multidisciplinary diagnosis report.'
)

# Create the Crew
crew = Crew(
    agents=[cardiologist, psychologist, pulmonologist, team_agent],
    tasks=[task_cardiologist, task_psychologist, task_pulmonologist, task_team_diagnosis],
    process=Process.sequential
)

# Execute the workflow
final_diagnosis = crew.kickoff(inputs={'medical_report': medical_report})

print(final_diagnosis)
