from abc import ABC, abstractmethod

from .context import ETLContext


class ETLStep(ABC):
    """
    Base class for all ETL pipeline steps with state management.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the step (e.g., 'fetch', 'parse')."""
        pass
    
    def execute(self, ctx: ETLContext) -> None:
        """
        Orchestration logic: checks state, runs step, updates state.
        """
        article_id = ctx.article.id
        state_db = ctx.state_manager
        artifacts_db = ctx.artifact_store
        
        # These steps don't save any artifacts so they always have to be called
        always_failed_steps = ["chunking", "embed"]
        
        current_status = state_db.get_status(article_id, self.name)
        
        if current_status == "done":
            print(f"Skipping step '{self.name}' for {article_id} (already done).")
            return
        
        print(f"Starting step '{self.name}' for {article_id}...")
        state_db.set_status(article_id, self.name, "running")
        
        try:
            if self.name in always_failed_steps:
                self.run(ctx)
                state_db.set_status(article_id, self.name, "failed")
                print(f"Step '{self.name}' completed for {article_id}.")
            else:
                self.run(ctx)
                state_db.set_status(article_id, self.name, "done")
                print(f"Step '{self.name}' completed for {article_id}.")
        
        except Exception as e:
            print(f"Step '{self.name}' failed for article {article_id}: {e}")
            state_db.set_status(article_id, self.name, "failed", error=str(e))
            artifacts_db.delete_step(article_id, self.name)
            raise e
    
    @abstractmethod
    def run(self, ctx: ETLContext) -> None:
        """
        Implementation of the specific step logic.
        Must use ctx.artifact_store to load input and save output.
        """
        pass
