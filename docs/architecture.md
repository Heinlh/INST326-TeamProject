# Research Project Management - Architecture Diagram

## Complete System Architecture

```
┌───────────────────────────────────────────────────────┐
│            ABSTRACT BASE LAYER                        │
│       (Defines Interface Contracts)                   │
├───────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────────────────────┐                   │
│  │ AbstractExperiment             │                   │
│  ├────────────────────────────────┤                   │
│  │ + experiment_id                │                   │
│  │ + title, dataset               │                   │
│  │                                │                   │
│  │ @abstractmethod                │                   │
│  │ + access_policy()              │                   │
│  │ + processs()                   │                   │
│  │ + get_drainage_type()          │                   │
│  └────────────────────────────────┘                   │
│             ▲                                         │
│             │                                         │
│             │ inherits                                │
│             │                                         │
└─────────────┼─────────────────────────────────────────┘
              │                                  │
┌─────────────┼──────────────────────────────────┼───────────────────────┐
│             │         CONCRETE IMPLEMENTATION LAYER                    │
│             │         (Inheritance & Polymorphism)                     │
├─────────────┴──────────────────────────────────┴───────────────────────┤
│                                                                        │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐     │
│  │   LabExperiment │  │ FieldStudy   │  │ Survey.                │     │
│  ├─────────────────┤  ├──────────────┤  ├────────────────────────┤     │
│  │+ biosafety_level│  │ + region.    │  │ + consent_col          │     │
│  │                 │  │              │  │                        │     │
│  │                 │  │              │  │                        │     │
│  │ access_policy() │  │              │  │                        │     │
│  │   → return      │  │ access_      │  │ access_policy ()       │     │
│  │     statement   │  │   policy()   │  │   → return statement   │     │
│  │ process()       │  │   → return   │  │                        │     │
│  │   → dataset.    │  │   statement  │  │ process().             │     │
│  └─────────────────┘  │ process()    │  │   → dataset.           │     │
│                       │   →  schema
                              dataset  │  │                        │     │
│                       └──────────────┘  │                        │     │
│                                         └────────────────────────┘     │
│                                                                        │
│                                                                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ used by
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPOSITION LAYER                                 │
│                   (System Coordination)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      ResearchProject                            │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │ + project_id, titles                                            │    │
│  │ + experiment []  ◄───── HAS-MANY experiments                    │    │
│  │ + policies []  ◄──────── HAS-MANY access_policies               │    │
│  │ + datasets [] ◄──────── HAS-MANY datasets                       │    │
│  │                                                                 │    │
│  │ + add_experiment(AbstractExperiment)                            │    │
│  │ + run_all()                                                     │    │
│  │ + policies()                                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Relationships

### INHERITANCE (is-a)
- LabExperiment **IS-AN** Experiment
- FieldStudy **IS-AN** Experiment
- Survey **IS-AN** Experiment


### COMPOSITION (has-a / has-many)
- ResearchProject **HAS-MANY** Experiments
- ResearchProject **HAS-MANY** Datasets


### POLYMORPHISM (same method, different behavior)
- All experiments implement `policies` differently:
  - FieldStudy returns "Research team + IRB-approved collaborators; de-identify GPS traces; share summaries externally."
  - LabExperiment returns "Restricted to approved lab members (BSL-{self.biosafety_level}) and PI; publish aggregated results only."
  - Survey returns ""PII restricted; aggregate results public; raw responses only to IRB-approved team."

- All experiments implement `process()` differently:
  - FieldStudy incorporates the time and region of the experiment
  - LabExperiment incorporates quanatative values
  - Survey incorporates and considers the ethics of the experiment

## Data Flow Example

```
1. User creates ResearchProject
   └─→ Manager initializes empty experiments[], policy[], datasets[]

2. User adds experiments to manager
   └─→ ResarchProject.add_experiment(LabExperiment)
   └─→ ResearchProject.add_experiment(Survey)


3. User inputs essential inforamtion 
   └─→ AbstractExperiment 
       └─→ experiment_id
       └─→ dataset  
       └─→ title     

4. User checks garden status
   └─→ Manager.get_garden_summary()
       └─→ Gives the access_policy of the experiment
       └─→ Returns dataset as advised
```

## Why This Architecture Works

### Abstract Layer Benefits
✓ Enforces consistent interfaces
✓ Cannot instantiate incomplete classes
✓ Clear contracts for implementers
✓ Type safety through ABC module

### Concrete Layer Benefits
✓ Code reuse through inheritance
✓ Specialized behavior per type
✓ Polymorphic method calls
✓ Easy to extend with new types

### Composition Layer Benefits
✓ Flexible object relationships
✓ No artificial inheritance hierarchies
✓ Clear separation of concerns
✓ System-level coordination

## Design Pattern Summary

| Pattern | Where Used | Why |
|---------|-----------|-----|
| **Abstract Base Class** | AbstractExperiment | Enforce interface contracts |
| **Inheritance** | Experiment hierarchies | Code reuse + polymorphism |
| **Polymorphism** | All calculate/get methods | Type-specific behavior |
| **Composition** | ResearchProject | Flexible relationships |
| **Template Method** | Abstract methods | Define algorithm structure |

