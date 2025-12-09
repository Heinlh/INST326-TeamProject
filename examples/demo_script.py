"""
Research Project Pipeline - Interactive Demo
Demonstrates the complete system working together
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiment import ResearchProject
#from plant import Vegetable, Herb, Flower



def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """Run the research project tracker demo."""
    
    print_section("Welcome to the Research Project Pipeline Demo")
    print("\nThis demo shows all OOP concepts in action:")
    print("  • Basic classes and encapsulation")
    print("  • Inheritance and polymorphism")
    print("  • Abstract base classes")
    print("  • Composition")
    print("  • System integration")
    
    # ========================================================================
    # Step 1: Initialize the Pipeline
    # ========================================================================
    print_section("Step 1: Initialize Research Project Pipeline")
    

    rsd = ResearchProject(
         project_id = "001-20251209",
         title = "Demo Project V1"
     )
    
    # print(f"\n{rsd}")
    # print(f"Season details: {garden.season}")
    
    # ========================================================================
    # Step 2: Create Containers (Demonstrating Inheritance)
    # ========================================================================
#     print_section("Step 2: Create Garden Containers (Inheritance)")
    
#     # Create a raised bed
#     vegetable_bed = garden.create_bed(
#         length=48, width=48, depth=12,
#         name="Main Vegetable Bed",
#         location="backyard_south",
#         is_raised=True
#     )
#     print(f"\nCreated: {vegetable_bed}")
#     print(f"  Area: {vegetable_bed.calculate_area()} sq in ({vegetable_bed.calculate_area()/144:.1f} sq ft)")
#     print(f"  Volume: {vegetable_bed.calculate_volume_gallons():.1f} gallons")
#     print(f"  Border material needed: {vegetable_bed.calculate_border_material():.1f} linear feet")
#     print(f"  Soil bags needed: {vegetable_bed.calculate_soil_bags_needed()} bags")
    
#     # Add a pot (demonstrates polymorphism - different area calculation)
#     herb_pot = garden.add_container(
#         garden.create_bed.__self__.__class__.__bases__[0](  # Get Pot class
#             container_type="pot", length=16, width=16, depth=14,
#             name="Herb Pot", location="patio"
#         )
#     )
    
#     # Note: For demo, we'll manually import and create specialized containers
#     from specialized_containers import Pot, Planter
    
#     herb_pot = Pot(diameter=16, depth=14, name="Herb Pot", location="patio")
#     garden.add_container(herb_pot)
#     print(f"\nCreated: {herb_pot}")
#     print(f"  Area: {herb_pot.calculate_area():.1f} sq in")
#     print(f"  Volume: {herb_pot.calculate_volume_gallons():.1f} gallons")
#     mobility = herb_pot.assess_mobility()
#     print(f"  Mobility: {mobility['estimated_weight_lbs']} lbs - {mobility['mobility_assessment']}")
    
#     # Add an accessible planter
#     flower_planter = Planter(
#         length=36, width=12, depth=8,
#         name="Accessible Flower Planter",
#         location="front_porch",
#         has_legs=True, leg_height=30
#     )
#     garden.add_container(flower_planter)
#     print(f"\nCreated: {flower_planter}")
#     accessibility = flower_planter.assess_accessibility()
#     print(f"  Total height: {flower_planter.total_height} inches")
#     print(f"  Accessibility: {accessibility['accessibility_rating']}")
#     print(f"  Wheelchair accessible: {accessibility['suitable_for_limited_mobility']}")
    
#     # ========================================================================
#     # Step 3: Create Cells (Demonstrating Composition)
#     # ========================================================================
#     print_section("Step 3: Create Planting Cells (Composition)")
    
#     # Create 4x4 grid for the raised bed (square foot gardening)
#     veg_cells = garden.create_cells_for_container(vegetable_bed, rows=4, columns=4)
#     print(f"\nCreated 4x4 grid in '{vegetable_bed.name}':")
#     print(f"  Total cells: {len(veg_cells)}")
#     print(f"  Cell size: 12\" x 12\" (square foot)")
#     print(f"  Cell locations: A1-A4, B1-B4, C1-C4, D1-D4")
    
#     # Create cells for other containers
#     herb_cells = garden.create_cells_for_container(herb_pot, rows=1, columns=1)
#     flower_cells = garden.create_cells_for_container(flower_planter, rows=1, columns=3)
    
#     print(f"\nTotal garden capacity: {len(veg_cells) + len(herb_cells) + len(flower_cells)} cells")
    
#     # ========================================================================
#     # Step 4: Create Plants (Demonstrating Abstract Base Classes & Polymorphism)
#     # ========================================================================
#     print_section("Step 4: Create Plant Library (Abstract Base Classes)")
    
#     print("\nCreating diverse plant types (polymorphism in action):")
    
#     # Vegetables
#     plants = [
#         Vegetable("Tomato", "Cherokee Purple", spacing_inches=24, days_to_maturity=80),
#         Vegetable("Lettuce", "Buttercrunch", spacing_inches=6, days_to_maturity=55),
#         Vegetable("Carrot", "Danvers", spacing_inches=3, days_to_maturity=70),
#         Vegetable("Pepper", "California Wonder", spacing_inches=18, days_to_maturity=75),
#         Herb("Basil", "Genovese", spacing_inches=12, days_to_usable=60),
#         Herb("Parsley", "Italian Flat Leaf", spacing_inches=8, days_to_usable=70),
#         Flower("Zinnia", "State Fair Mix", spacing_inches=12, days_to_bloom=60, 
#                bloom_duration_days=7, color="mixed"),
#         Flower("Marigold", "French Vanilla", spacing_inches=10, days_to_bloom=50,
#                bloom_duration_days=14, color="cream")
#     ]
    
#     for plant in plants:
#         garden.add_plant_to_library(plant)
#         print(f"  Added {plant.__class__.__name__}: {plant.name} '{plant.variety}'")
#         print(f"    Days to maturity: {plant.get_days_to_maturity()}")
#         care = plant.get_care_requirements()
#         print(f"    Water needs: {care['water']}")
    
#     # ========================================================================
#     # Step 5: Plant the Garden (System Integration)
#     # ========================================================================
#     print_section("Step 5: Plant the Garden (System Integration)")
    
#     # Simulate a spring planting date (2 weeks after last frost)
#     planting_date = garden.season.last_frost_date + timedelta(weeks=2)
#     print(f"\nPlanting date: {planting_date.strftime('%B %d, %Y')}")
#     print(f"  (Safe planting: 2 weeks after last frost)")
    
#     # Plant vegetables in the bed
#     planting_plan = [
#         ("A1", "Tomato", "Cherokee Purple"),
#         ("A4", "Tomato", "Cherokee Purple"),
#         ("B2", "Lettuce", "Buttercrunch"),
#         ("B3", "Lettuce", "Buttercrunch"),
#         ("C1", "Basil", "Genovese"),  # Companion plant for tomatoes
#         ("C2", "Carrot", "Danvers"),
#         ("C3", "Carrot", "Danvers"),
#         ("D2", "Pepper", "California Wonder"),
#     ]
    
#     print("\nPlanting vegetables:")
#     for cell_loc, plant_name, variety in planting_plan:
#         cell = garden.get_cell_by_location("Main Vegetable Bed", cell_loc)
#         plant = garden.get_plant_from_library(plant_name, variety)
        
#         # Create a new instance for actual planting
#         if isinstance(plant, Vegetable):
#             new_plant = Vegetable(plant.name, plant.variety, 
#                                 plant.spacing_inches, plant.get_days_to_maturity())
#         elif isinstance(plant, Herb):
#             new_plant = Herb(plant.name, plant.variety,
#                            plant.spacing_inches, plant.get_days_to_maturity())
#         else:
#             new_plant = Flower(plant.name, plant.variety,
#                              plant.spacing_inches, plant.get_days_to_maturity(),
#                              plant.bloom_duration_days, plant.color)
        
#         new_plant.plant(planting_date)
#         success = garden.plant_in_cell(cell, new_plant)
#         if success:
#             print(f"  ✓ {cell_loc}: {plant_name} '{variety}'")
    
#     # Plant herbs in pot
#     basil = Herb("Basil", "Genovese", 12, 60)
#     basil.plant(planting_date)
#     garden.plant_in_cell(herb_cells[0], basil)
#     print(f"\n  ✓ Herb Pot: Basil 'Genovese'")
    
#     # Plant flowers in planter
#     flower_planting = [
#         ("A1", "Marigold", "French Vanilla"),
#         ("A2", "Zinnia", "State Fair Mix"),
#         ("A3", "Marigold", "French Vanilla"),
#     ]
    
#     print("\nPlanting flowers:")
#     for cell_loc, plant_name, variety in flower_planting:
#         cell = garden.get_cell_by_location("Accessible Flower Planter", cell_loc)
#         plant = garden.get_plant_from_library(plant_name, variety)
#         new_plant = Flower(plant.name, plant.variety, plant.spacing_inches,
#                           plant.get_days_to_maturity(), 7, plant.color)
#         new_plant.plant(planting_date)
#         garden.plant_in_cell(cell, new_plant)
#         print(f"  ✓ {cell_loc}: {plant_name} '{variety}'")
    
#     # ========================================================================
#     # Step 6: View Garden Status
#     # ========================================================================
#     print_section("Step 6: Garden Status and Analytics")
    
#     summary = garden.get_garden_summary()
#     print("\nGarden Summary:")
#     print(f"  Total containers: {summary['total_containers']}")
#     print(f"  Total cells: {summary['total_cells']}")
#     print(f"  Occupied cells: {summary['occupied_cells']}")
#     print(f"  Empty cells: {summary['empty_cells']}")
#     print(f"  Total growing area: {summary['total_growing_area_sqft']:.1f} sq ft")
#     print(f"  Frost status: {summary['frost_status']}")
    
#     # ========================================================================
#     # Step 7: Planting Schedule
#     # ========================================================================
#     print_section("Step 7: Planting Schedule")
    
#     schedule = garden.get_planting_schedule()
#     print(f"\nPlanting schedule ({len(schedule)} plants):")
#     print(f"{'Container':<25} {'Location':<10} {'Plant':<30} {'Days Old':<10} {'Days to Mature':<15}")
#     print("-" * 100)
    
#     for item in schedule[:10]:  # Show first 10
#         print(f"{item['container']:<25} {item['location']:<10} {item['plant']:<30} "
#               f"{item['days_since_planting']:<10} {item['days_to_maturity'] or 'N/A':<15}")
    
#     # ========================================================================
#     # Step 8: Companion Planting Analysis
#     # ========================================================================
#     print_section("Step 8: Companion Planting Analysis")
    
#     companion_report = garden.get_companion_planting_report()
#     if companion_report:
#         print("\nCompanion planting relationships:")
#         for item in companion_report:
#             status = "✓ Compatible" if item['compatible'] else "⚠ Incompatible"
#             print(f"  {status}: {item['plant1']} & {item['plant2']}")
#             print(f"    Location: {item['cell1']} and {item['cell2']}")
#             print(f"    {item['recommendation']}")
#     else:
#         print("\nNo companion planting relationships detected")
    
#     # ========================================================================
#     # Step 9: Upcoming Tasks
#     # ========================================================================
#     print_section("Step 9: Upcoming Tasks")
    
#     # Simulate advancing time to show upcoming harvests
#     print("\n(Simulating 60 days of growth...)")
#     tasks = garden.get_upcoming_tasks(days_ahead=30)
    
#     if tasks:
#         print(f"\nUpcoming tasks in next 30 days:")
#         for task in tasks:
#             print(f"  {task['type'].upper()} - In {task['days_until']} days")
#             print(f"    Location: {task['location']}")
#             print(f"    Plant: {task['plant']}")
#             print(f"    {task['description']}")
#     else:
#         print("\nNo urgent tasks in the next 30 days")
    
#     # ========================================================================
#     # Step 10: Export Garden Data
#     # ========================================================================
#     print_section("Step 10: Data Export")
    
#     export_data = garden.export_garden_data()
#     print("\nGarden data export includes:")
#     print(f"  • Garden configuration")
#     print(f"  • {len(export_data['containers'])} containers with full specifications")
#     print(f"  • {len(export_data['plantings'])} active plantings")
#     print(f"  • {len(export_data['log'])} log entries")
#     print(f"\nData can be saved to JSON for backup or analysis")
    
#     # ========================================================================
#     # Conclusion
#     # ========================================================================
#     print_section("Demo Complete!")
    
#     print("\nThis demo demonstrated:")
#     print("  ✓ Encapsulation - Private attributes with properties")
#     print("  ✓ Inheritance - Bed, Pot, Planter from PlantingContainer")
#     print("  ✓ Polymorphism - Different area calculations, plant types")
#     print("  ✓ Abstract Base Classes - Plant with concrete implementations")
#     print("  ✓ Composition - Cell HAS-A PlantingContainer")
#     print("  ✓ Class methods - Season utility methods")
#     print("  ✓ System integration - GardenManager coordinating everything")
    
#     print("\nThe garden system is ready for your adaptations!")
#     print("Try creating similar systems for:")
#     print("  • Library Management (Book, Member, Loan, Catalog)")
#     print("  • Research Data (Experiment, Dataset, Analysis, Researcher)")
#     print("  • Digital Archives (Document, Collection, User, SearchIndex)")
    
#     print("\n" + "="*70)


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\n❌ Error running demo: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
