from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        return list(self._modules.values())

    def parameters(self) -> List[Parameter]:
        return list(self._parameters.values())

    def add_parameter(self, name: str, parameter: Parameter) -> None:
        self._parameters[name] = parameter

    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Forward method not implemented")

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.forward(*inputs, **kwargs)

    def train(self) -> None:
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        out = []
        for name, param in self._parameters.items():
            out.append((name, param))
        for name, module in self._modules.items():
            for sub_name, param in module.named_parameters():
                out.append((f"{name}.{sub_name}", param))
        return out


class Parameter:
    def __init__(self, value: float) -> None:
        self.value = value
        self.grad = 0.0
