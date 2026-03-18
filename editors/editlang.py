"""EditLang - PDDL-inspired action specification language for scene editing.

Defines the core data structures for actions, predicates, and domains.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Any, Optional
import json
import yaml

# Type alias for predicates: ("predicate_name", ("arg1", "arg2", ...))
Predicate = Tuple[str, Tuple[str, ...]]


@dataclass
class Action:
    """Represents a single action in the EditLang domain.
    
    Attributes:
        name: Action name (e.g., "rotate_towards")
        args: Action arguments as a dict (e.g., {"obj": "chair_01", "anchor": "window_01"})
        pre: Set of preconditions that must hold before action
        add: Set of predicates added after action execution
        dele: Set of predicates deleted after action execution (avoiding 'del' keyword)
    """
    name: str
    args: Dict[str, Any]
    pre: Set[Predicate] = field(default_factory=set)
    add: Set[Predicate] = field(default_factory=set)
    dele: Set[Predicate] = field(default_factory=set)  # "del" is reserved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {
            "name": self.name,
            "args": self.args,
            "pre": [{"pred": p[0], "args": list(p[1])} for p in self.pre],
            "add": [{"pred": p[0], "args": list(p[1])} for p in self.add],
            "dele": [{"pred": p[0], "args": list(p[1])} for p in self.dele],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary."""
        pre = {(p["pred"], tuple(p["args"])) for p in data.get("pre", [])}
        add = {(p["pred"], tuple(p["args"])) for p in data.get("add", [])}
        dele = {(p["pred"], tuple(p["args"])) for p in data.get("dele", [])}
        return cls(
            name=data["name"],
            args=data.get("args", {}),
            pre=pre,
            add=add,
            dele=dele
        )


@dataclass
class EditLangDomain:
    """Represents the domain of available actions.
    
    Attributes:
        actions: Dictionary mapping action names to Action schemas
        predicates: Optional set of valid predicate names
    """
    actions: Dict[str, Action] = field(default_factory=dict)
    predicates: Optional[Set[str]] = None
    
    def add_action(self, action: Action) -> None:
        """Add an action to the domain."""
        self.actions[action.name] = action
    
    def get_action(self, name: str) -> Optional[Action]:
        """Get action by name."""
        return self.actions.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert domain to dictionary for serialization."""
        return {
            "actions": {name: action.to_dict() for name, action in self.actions.items()},
            "predicates": list(self.predicates) if self.predicates else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditLangDomain":
        """Create Domain from dictionary."""
        actions = {
            name: Action.from_dict(action_data)
            for name, action_data in data.get("actions", {}).items()
        }
        predicates = set(data["predicates"]) if data.get("predicates") else None
        return cls(actions=actions, predicates=predicates)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "EditLangDomain":
        """Load domain from YAML file with validation."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create domain first
        domain = cls.from_dict(data)
        
        # Validate that all predicates used in actions are defined
        if domain.predicates:
            for action_name, action in domain.actions.items():
                # Check predicates in pre, add, dele
                for pred in action.pre | action.add | action.dele:
                    pred_name = pred[0]
                    if pred_name not in domain.predicates and not pred_name.startswith('?'):
                        raise ValueError(f"Undefined predicate '{pred_name}' in action '{action_name}'")
        
        return domain
    
    def to_yaml(self, filepath: str) -> None:
        """Save domain to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def standard_domain() -> EditLangDomain:
    """Factory function to create standard domain from YAML specification.
    
    Returns:
        EditLangDomain loaded from editors/editlang_std.yaml
    """
    import os
    from pathlib import Path
    
    # Find the YAML file relative to this module
    module_dir = Path(__file__).parent
    yaml_path = module_dir / "editlang_std.yaml"
    
    # Load and return the domain
    return EditLangDomain.from_yaml(str(yaml_path))


def instantiate_action(schema: Action, args: Dict[str, Any]) -> Action:
    """Instantiate an action schema with specific arguments.
    
    Args:
        schema: Action schema template
        args: Arguments to bind to the action
        
    Returns:
        New Action instance with bound arguments
    """
    # Replace variables in predicates with actual values
    def bind_predicate(pred: Predicate) -> Predicate:
        pred_name, pred_args = pred
        bound_args = tuple(
            args.get(arg, arg) for arg in pred_args
        )
        return (pred_name, bound_args)
    
    return Action(
        name=schema.name,
        args=args,
        pre={bind_predicate(p) for p in schema.pre},
        add={bind_predicate(p) for p in schema.add},
        dele={bind_predicate(p) for p in schema.dele}
    )
