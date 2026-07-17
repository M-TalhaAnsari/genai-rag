import os
from dotenv import load_dotenv
import yaml
from google import genai
from google.genai import types
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from src.models import RecipeSuggestionOutput, NutrientAnalysisOutput
from src.tools import (
    ExtractIngredientsTool,
    FilterIngredientsTool,
    DietaryFilterTool,
    NutrientAnalysisTool
)

load_dotenv()

# Raw GenAI client — used directly inside src/tools.py for the actual
# vision / nutrition calls. Kept separate from the LLM object below, which
# is what CrewAI's agents use for their own reasoning/tool-orchestration.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# CrewAI's own reasoning LLM. Without this, agents default to OpenAI, which
# needs a separate billed API key. Pointing them at Gemini instead means
# both the agent reasoning and the tool calls run on the same Gemini key.
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Get the absolute path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


@CrewBase
class BaseNourishBotCrew:
    agents_config_path = os.path.join(CONFIG_DIR, "agents.yaml")
    tasks_config_path = os.path.join(CONFIG_DIR, "tasks.yaml")

    def __init__(self, image_data, dietary_restrictions: str = None) -> None:
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions

        with open(self.agents_config_path, "r") as f:
            self.agents_config = yaml.safe_load(f)
        with open(self.tasks_config_path, "r") as f:
            self.tasks_config = yaml.safe_load(f)

    @agent
    def ingredient_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_detection_agent'],
            tools=[
                ExtractIngredientsTool.extract_ingredients_from_image,
                FilterIngredientsTool.filter_ingredients
            ],
            llm=gemini_llm,
            allow_delegation=False,
            max_iter=2,
            verbose=True
        )

    @agent
    def dietary_filtering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['dietary_filtering_agent'],
            tools=[DietaryFilterTool.dietary_filter],
            llm=gemini_llm,
            allow_delegation=False,
            max_iter=2,
            verbose=True
        )

    @agent
    def nutrient_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrient_analysis_agent'],
            tools=[NutrientAnalysisTool.analyze_image],
            llm=gemini_llm,
            allow_delegation=False,
            max_iter=2,
            verbose=True
        )

    @agent
    def recipe_suggestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['recipe_suggestion_agent'],
            tools=[],
            llm=gemini_llm,
            allow_delegation=False,
            max_iter=2,
            verbose=True
        )

    @task
    def ingredient_detection_task(self) -> Task:
        task_config = self.tasks_config['ingredient_detection_task']
        return Task(
            description=task_config['description'],
            agent=self.ingredient_detection_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def dietary_filtering_task(self) -> Task:
        task_config = self.tasks_config['dietary_filtering_task']

        return Task(
            description=task_config['description'],
            agent=self.dietary_filtering_agent(),
            depends_on=['ingredient_detection_task'],
            input_data=lambda outputs: {
                'ingredients': outputs['ingredient_detection_task'],
                'dietary_restrictions': self.dietary_restrictions
            },
            expected_output=task_config['expected_output']
        )

    @task
    def nutrient_analysis_task(self) -> Task:
        task_config = self.tasks_config['nutrient_analysis_task']

        return Task(
            description=task_config['description'],
            agent=self.nutrient_analysis_agent(),
            expected_output=task_config['expected_output'],
            output_json=NutrientAnalysisOutput
        )

    @task
    def recipe_suggestion_task(self) -> Task:
        task_config = self.tasks_config['recipe_suggestion_task']

        return Task(
            description=task_config['description'],
            agent=self.recipe_suggestion_agent(),
            depends_on=['dietary_filtering_task', 'nutrient_analysis_task'],
            input_data=lambda outputs: {
                'filtered_ingredients': outputs['dietary_filtering_task'],
                'nutrient_analysis': outputs['nutrient_analysis_task']
            },
            expected_output=task_config['expected_output'],
            output_json=RecipeSuggestionOutput
        )


@CrewBase
class NourishBotRecipeCrew(BaseNourishBotCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [
            self.ingredient_detection_task(),
            self.dietary_filtering_task(),
            self.recipe_suggestion_task()
        ]
        agents = [
            self.ingredient_detection_agent(),
            self.dietary_filtering_agent(),
            self.recipe_suggestion_agent()
        ]

        return Crew(
            name="NourishBotRecipeCrew",
            tasks=tasks,
            agents=agents,
            process=Process.sequential,
            verbose=True,
            max_rpm=4,  # stay under Gemini free-tier's 5 req/min cap on gemini-2.5-flash
        )


@CrewBase
class NourishBotAnalysisCrew(BaseNourishBotCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [
            self.nutrient_analysis_task()
        ]
        agents = [
            self.nutrient_analysis_agent()
        ]

        return Crew(
            name="NourishBotAnalysisCrew",
            tasks=tasks,
            agents=agents,
            process=Process.sequential,
            verbose=True,
            max_rpm=4,  
        )