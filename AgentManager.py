"""
AgentManager - Unified orchestration layer for LangGraph workflows.

This module provides a single entry point for all agent operations,
handling workflow selection, state initialization, and result normalization.
"""

from enum import Enum
from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph

from State import AgentState, DocumentIngestState
from Agent import graph as query_graph
from DocumentIngestAgent import document_ingest_graph


# ============================================================================
# Enumerations
# ============================================================================

class WorkflowType(str, Enum):
    """Types of workflows available in the system."""
    QUERY = "query"
    INGEST = "ingest"

    @classmethod
    def from_request(cls, **kwargs) -> 'WorkflowType':
        """
        Automatically determine workflow type from request parameters.

        Args:
            **kwargs: Request parameters

        Returns:
            WorkflowType: Detected workflow type

        Raises:
            ValueError: If workflow type cannot be determined
        """
        # Check for explicit workflow_type parameter
        if "workflow_type" in kwargs:
            return cls(kwargs["workflow_type"])

        # Check for question/query (Query workflow)
        if "question" in kwargs or "query" in kwargs or "messages" in kwargs:
            return cls.QUERY

        # Check for file_path/document (Ingest workflow)
        if "file_path" in kwargs or "document" in kwargs or "source_filename" in kwargs:
            return cls.INGEST

        raise ValueError(
            "Could not determine workflow type from request parameters. "
            "Provide 'workflow_type', 'question', or 'file_path'."
        )


class ExecutionStatus(str, Enum):
    """Standardized execution status codes."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


# ============================================================================
# Result Models
# ============================================================================

@dataclass
class WorkflowResult:
    """
    Standardized result container for all workflows.

    This provides a unified interface regardless of which workflow was executed.
    """
    workflow_type: WorkflowType
    status: ExecutionStatus
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "workflow_type": self.workflow_type.value,
            "status": self.status.value,
            "success": self.success,
            "data": self.data
        }

        if self.error:
            result["error"] = self.error

        if self.metadata:
            result["metadata"] = self.metadata

        return result


# ============================================================================
# Workflow Strategy Pattern
# ============================================================================

class BaseWorkflow(ABC):
    """
    Abstract base class for workflow strategies.

    Each workflow type implements this interface to provide
    consistent initialization and execution patterns.
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def initialize_state(self, **kwargs) -> Union[AgentState, DocumentIngestState]:
        """
        Initialize state from request parameters.

        Args:
            **kwargs: Request parameters

        Returns:
            Initialized state object
        """
        pass

    @abstractmethod
    def execute(self, state: Union[AgentState, DocumentIngestState]) -> Dict[str, Any]:
        """
        Execute the workflow with given state.

        Args:
            state: Initialized state object

        Returns:
            Raw workflow result
        """
        pass

    @abstractmethod
    def normalize_result(self, raw_result: Dict[str, Any]) -> WorkflowResult:
        """
        Convert workflow-specific result to standardized format.

        Args:
            raw_result: Raw result from workflow execution

        Returns:
            Standardized WorkflowResult
        """
        pass

    def run(self, **kwargs) -> WorkflowResult:
        """
        Full workflow execution pipeline.

        Args:
            **kwargs: Request parameters

        Returns:
            Standardized WorkflowResult
        """
        try:
            # Initialize state
            state = self.initialize_state(**kwargs)
            self.logger.info(f"Initialized state for {self.__class__.__name__}")

            # Execute workflow
            raw_result = self.execute(state)
            self.logger.info(f"Executed {self.__class__.__name__}")

            # Normalize result
            result = self.normalize_result(raw_result)
            self.logger.info(f"Normalized result: {result.status}")

            return result

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return WorkflowResult(
                workflow_type=self._get_workflow_type(),
                status=ExecutionStatus.ERROR,
                data={},
                error=str(e)
            )

    @abstractmethod
    def _get_workflow_type(self) -> WorkflowType:
        """Get the workflow type for this strategy."""
        pass


class QueryWorkflow(BaseWorkflow):
    """Strategy for query/retrieval workflows."""

    def _get_workflow_type(self) -> WorkflowType:
        return WorkflowType.QUERY

    def initialize_state(self, **kwargs) -> AgentState:
        """
        Initialize AgentState from query parameters.

        Expected kwargs:
            - question (str): The user's question
            OR
            - messages (List[BaseMessage]): Pre-formatted messages

        Returns:
            AgentState: Initialized state
        """
        # Get question from various possible keys
        question = kwargs.get("question") or kwargs.get("query")
        messages = kwargs.get("messages")

        if not messages and not question:
            raise ValueError("Either 'question' or 'messages' must be provided")

        # Create messages if not provided
        if not messages:
            messages = [HumanMessage(content=question)]

        return {
            "messages": messages,
            "routing_decision": ""
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute the query graph."""
        return self.graph.invoke(state)

    def normalize_result(self, raw_result: Dict[str, Any]) -> WorkflowResult:
        """
        Normalize query workflow result.

        Extracts:
            - answer: Final LLM response
            - routing: Which specialist agent was used
            - db_results_count: Number of relevant chunks found
            - question: Original question
        """
        try:
            messages = raw_result.get("messages", [])
            answer = messages[-1].content if messages else "No answer generated"

            data = {
                "answer": answer,
                "routing": raw_result.get("routing_decision", "unknown"),
                "db_results_count": len(raw_result.get("db_results", [])),
                "question": raw_result.get("query", "")
            }

            metadata = {
                "message_count": len(messages),
                "db_results": raw_result.get("db_results", [])[:3]  # First 3 for debugging
            }

            return WorkflowResult(
                workflow_type=WorkflowType.QUERY,
                status=ExecutionStatus.SUCCESS,
                data=data,
                metadata=metadata
            )

        except Exception as e:
            return WorkflowResult(
                workflow_type=WorkflowType.QUERY,
                status=ExecutionStatus.ERROR,
                data={},
                error=f"Failed to normalize query result: {str(e)}"
            )


class IngestWorkflow(BaseWorkflow):
    """Strategy for document ingestion workflows."""

    def _get_workflow_type(self) -> WorkflowType:
        return WorkflowType.INGEST

    def initialize_state(self, **kwargs) -> DocumentIngestState:
        """
        Initialize DocumentIngestState from upload parameters.

        Expected kwargs:
            - file_path (str): Path to uploaded PDF
            - source_filename (str): Original filename
            - max_length (int): Chunk size (default: 1000)
            - overlap (int): Chunk overlap (default: 100)

        Returns:
            DocumentIngestState: Initialized state
        """
        file_path = kwargs.get("file_path")
        source_filename = kwargs.get("source_filename")

        if not file_path or not source_filename:
            raise ValueError("Both 'file_path' and 'source_filename' are required")

        return {
            "file_path": file_path,
            "source_filename": source_filename,
            "max_length": kwargs.get("max_length", 1000),
            "overlap": kwargs.get("overlap", 100),
            "status": "validating",
            "current_step": "Validating PDF file",
            "chunks_stored": 0
        }

    def execute(self, state: DocumentIngestState) -> Dict[str, Any]:
        """Execute the document ingest graph."""
        return self.graph.invoke(state)

    def normalize_result(self, raw_result: Dict[str, Any]) -> WorkflowResult:
        """
        Normalize ingest workflow result.

        Extracts:
            - chunks_stored: Number of chunks successfully stored
            - status: Final processing status
            - current_step: Last completed step
        """
        try:
            status_value = raw_result.get("status", "unknown")

            # Map workflow status to ExecutionStatus
            if status_value == "complete":
                exec_status = ExecutionStatus.SUCCESS
            elif status_value == "error":
                exec_status = ExecutionStatus.ERROR
            else:
                exec_status = ExecutionStatus.PARTIAL

            data = {
                "chunks_stored": raw_result.get("chunks_stored", 0),
                "status": status_value,
                "current_step": raw_result.get("current_step", ""),
                "source_filename": raw_result.get("source_filename", "")
            }

            error = raw_result.get("error_message")

            metadata = {
                "extracted_pages": len(raw_result.get("extracted_pages", [])),
                "chunks_created": len(raw_result.get("chunks", []))
            }

            return WorkflowResult(
                workflow_type=WorkflowType.INGEST,
                status=exec_status,
                data=data,
                error=error,
                metadata=metadata
            )

        except Exception as e:
            return WorkflowResult(
                workflow_type=WorkflowType.INGEST,
                status=ExecutionStatus.ERROR,
                data={},
                error=f"Failed to normalize ingest result: {str(e)}"
            )


# ============================================================================
# Workflow Factory
# ============================================================================

class WorkflowFactory:
    """
    Factory for creating workflow strategy instances.

    Centralizes workflow instantiation and makes it easy to add
    new workflow types in the future.
    """

    _workflows: Dict[WorkflowType, BaseWorkflow] = {}

    @classmethod
    def register_workflow(cls, workflow_type: WorkflowType, workflow: BaseWorkflow):
        """
        Register a workflow strategy.

        Args:
            workflow_type: Type identifier
            workflow: Workflow strategy instance
        """
        cls._workflows[workflow_type] = workflow

    @classmethod
    def get_workflow(cls, workflow_type: WorkflowType) -> BaseWorkflow:
        """
        Get workflow strategy by type.

        Args:
            workflow_type: Type to retrieve

        Returns:
            BaseWorkflow: Workflow strategy instance

        Raises:
            ValueError: If workflow type not registered
        """
        if workflow_type not in cls._workflows:
            raise ValueError(
                f"Unknown workflow type: {workflow_type}. "
                f"Available: {list(cls._workflows.keys())}"
            )

        return cls._workflows[workflow_type]

    @classmethod
    def initialize_default_workflows(cls):
        """Initialize and register all default workflows."""
        cls.register_workflow(
            WorkflowType.QUERY,
            QueryWorkflow(graph=query_graph)
        )

        cls.register_workflow(
            WorkflowType.INGEST,
            IngestWorkflow(graph=document_ingest_graph)
        )


# ============================================================================
# AgentManager (Facade)
# ============================================================================

class AgentManager:
    """
    Unified facade for all agent workflows.

    This class provides a single entry point for executing any workflow
    in the system, automatically handling workflow selection, state
    initialization, and result normalization.

    Usage:
        # Automatic workflow detection
        result = AgentManager.execute(question="What is benzene?")

        # Explicit workflow type
        result = AgentManager.execute(
            workflow_type=WorkflowType.INGEST,
            file_path="/tmp/doc.pdf",
            source_filename="doc.pdf"
        )

        # Check result
        if result.success:
            print(result.data)
        else:
            print(result.error)
    """

    _initialized = False
    _logger = logging.getLogger(__name__)

    @classmethod
    def initialize(cls):
        """Initialize the AgentManager and all workflows."""
        if cls._initialized:
            return

        cls._logger.info("Initializing AgentManager...")
        WorkflowFactory.initialize_default_workflows()
        cls._initialized = True
        cls._logger.info("AgentManager initialized successfully")

    @classmethod
    def execute(cls, **kwargs) -> WorkflowResult:
        """
        Execute a workflow with automatic type detection.

        Args:
            workflow_type (Optional[WorkflowType]): Explicit workflow type
            **kwargs: Workflow-specific parameters

        Returns:
            WorkflowResult: Standardized result

        Raises:
            ValueError: If workflow type cannot be determined
        """
        # Ensure initialized
        cls.initialize()

        # Determine workflow type
        if "workflow_type" in kwargs:
            workflow_type = WorkflowType(kwargs.pop("workflow_type"))
        else:
            workflow_type = WorkflowType.from_request(**kwargs)

        cls._logger.info(f"Executing {workflow_type.value} workflow")

        # Get and execute workflow
        workflow = WorkflowFactory.get_workflow(workflow_type)
        result = workflow.run(**kwargs)

        cls._logger.info(
            f"Workflow {workflow_type.value} completed with status: {result.status}"
        )

        return result

    @classmethod
    def execute_query(cls, question: str, **kwargs) -> WorkflowResult:
        """
        Convenience method for query workflow.

        Args:
            question: User's question
            **kwargs: Additional query parameters

        Returns:
            WorkflowResult: Query result
        """
        return cls.execute(workflow_type=WorkflowType.QUERY, question=question, **kwargs)

    @classmethod
    def execute_ingest(
        cls,
        file_path: str,
        source_filename: str,
        max_length: int = 1000,
        overlap: int = 100,
        **kwargs
    ) -> WorkflowResult:
        """
        Convenience method for ingest workflow.

        Args:
            file_path: Path to PDF file
            source_filename: Original filename
            max_length: Chunk size
            overlap: Chunk overlap
            **kwargs: Additional ingest parameters

        Returns:
            WorkflowResult: Ingest result
        """
        return cls.execute(
            workflow_type=WorkflowType.INGEST,
            file_path=file_path,
            source_filename=source_filename,
            max_length=max_length,
            overlap=overlap,
            **kwargs
        )

    @classmethod
    def get_available_workflows(cls) -> List[str]:
        """Get list of available workflow types."""
        cls.initialize()
        return [wf.value for wf in WorkflowType]


# ============================================================================
# Initialize on module import
# ============================================================================

AgentManager.initialize()