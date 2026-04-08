import gradio as gr
from typing import Dict, Any
import sys
from types import ModuleType

# Mock the `openenv` module and `Rubric` base class locally so grading.py can be imported
mock_openenv = ModuleType("openenv")
mock_openenv_core = ModuleType("openenv.core")
mock_openenv_core_rubrics = ModuleType("openenv.core.rubrics")

class MockRubric:
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, action, observation):
        return self.forward(action, observation)

mock_openenv_core_rubrics.Rubric = MockRubric
sys.modules["openenv"] = mock_openenv
sys.modules["openenv.core"] = mock_openenv_core
sys.modules["openenv.core.rubrics"] = mock_openenv_core_rubrics

# Import the grader from our existing module
from grading import DynapriceTaskGraderSuite

class MockObservation:
    """A mock observation class to simulate environment inputs."""
    def __init__(
        self,
        raw_reward: float,
        completed_rides: float,
        demand: float,
        supply: float,
        surge_multiplier: float,
        done: bool,
        task_id: str = "easy"
    ):
        self.metadata = {
            "raw_reward": raw_reward,
            "completed_rides": completed_rides,
            "task_id": task_id
        }
        self.demand = demand
        self.supply = supply
        self.surge_multiplier = surge_multiplier
        self.done = done

def evaluate_grading(
    task_id: str,
    raw_reward: float,
    completed_rides: float,
    demand: float,
    supply: float,
    surge_multiplier: float,
    done: bool
) -> Dict[str, float]:
    """Evaluates the scores using DynapriceTaskGraderSuite."""
    grader = DynapriceTaskGraderSuite()
    
    # Create the dummy observation based on inputs
    observation = MockObservation(
        raw_reward=raw_reward,
        completed_rides=completed_rides,
        demand=demand,
        supply=supply,
        surge_multiplier=surge_multiplier,
        done=done,
        task_id=task_id
    )
    
    # Generate the task scores mapping
    scores = grader.task_scores(action=None, observation=observation)
    return scores

def build_gui():
    """Builds and launches the Gradio Application."""
    with gr.Blocks(title="Dynaprice Grader Visualizer") as app:
        gr.Markdown("# Dynaprice Grader Visualizer")
        gr.Markdown(
            "Use this interface to test how different observation values affect the Grading Score "
            "across the **easy**, **medium**, and **hard** tasks."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Observation Parameters")
                task_id = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task Level (task_id metadata)")
                raw_reward = gr.Number(value=0.0, label="Raw Reward", step=1.0)
                completed_rides = gr.Number(value=0.0, label="Completed Rides", step=1.0)
                demand = gr.Number(value=50.0, label="Demand", step=1.0)
                supply = gr.Number(value=40.0, label="Supply", step=1.0)
                surge_multiplier = gr.Number(value=1.5, label="Surge Multiplier", step=0.1)
                done = gr.Checkbox(value=False, label="Environment Terminal State (done)")
                
                evaluate_btn = gr.Button("Calculate Scores", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Evaluated Task Scores")
                scores_output = gr.JSON(label="Scores per Task Grader")

        evaluate_btn.click(
            fn=evaluate_grading,
            inputs=[task_id, raw_reward, completed_rides, demand, supply, surge_multiplier, done],
            outputs=[scores_output]
        )

    return app

if __name__ == "__main__":
    app = build_gui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
