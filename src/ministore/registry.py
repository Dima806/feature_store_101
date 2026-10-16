"""The registry: the shared catalog of what each feature means.

"One definition, used everywhere" is the structural idea behind a feature store,
not a bureaucratic one. An entity says what a row is keyed on; a feature says
what is computed and which entity/timestamp it belongs to; a feature view groups
features from one timestamped source. Both the training path and the serving
path look features up here, so they can never silently disagree about meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    """What a feature row is keyed on, e.g. a customer id."""

    name: str
    join_key: str
    description: str = ""


@dataclass(frozen=True)
class Feature:
    """A single feature: its name, owning entity, dtype and description."""

    name: str
    entity: str
    dtype: str = "int64"
    description: str = ""


@dataclass
class FeatureView:
    """A group of features sharing one timestamped source and entity."""

    name: str
    entity: str
    features: list[str]
    timestamp_col: str
    source: str = ""


@dataclass
class Registry:
    """A tiny in-memory catalog of entities, features and feature views."""

    entities: dict[str, Entity] = field(default_factory=dict)
    features: dict[str, Feature] = field(default_factory=dict)
    feature_views: dict[str, FeatureView] = field(default_factory=dict)

    def register_entity(self, entity: Entity) -> Entity:
        self.entities[entity.name] = entity
        return entity

    def register_feature(self, feature: Feature) -> Feature:
        if feature.entity not in self.entities:
            raise KeyError(f"unknown entity {feature.entity!r} for feature {feature.name!r}")
        self.features[feature.name] = feature
        return feature

    def register_feature_view(self, view: FeatureView) -> FeatureView:
        if view.entity not in self.entities:
            raise KeyError(f"unknown entity {view.entity!r} for view {view.name!r}")
        self.feature_views[view.name] = view
        return view

    def join_key(self, feature: str) -> str:
        """The entity join key a feature is keyed on."""
        entity_name = self.features[feature].entity
        return self.entities[entity_name].join_key
