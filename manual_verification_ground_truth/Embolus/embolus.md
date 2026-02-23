# Manual verification: Embolus

We manually checked the **parent**, **grandparent**, **children**, and **sibling** relationships for this concept.

## Parents

![Parents](parents.png)

## Grandparents

![Grandparents](grandparents.png)

## Childrens

![Childrens](children.png)

## Siblings

![Siblings 1st half](sibling1.png)
![Siblings 2nd half](sibling2.png)




# -----------------------------
# Define Set 1
# -----------------------------
set1 = {
    "Accumulation of fluid", "Acquired absence", "Acquired web", "Aneurysm", "Ankylosis", "Blood clot", "Bulging", "Calculus", "Canalization", "Cavity", "Collapse", "Compression", "Cyst", "Disintegration", "Disruption", "Dissecting hemorrhage", "Diverticulum", "Ecchymosis", "Entrapment", "Failure of fusion", "False passage", "Fenestration", "Fibrin strand", "Fissure", "Fistula", "Fracture", "Fragmentation", "Hematocele", "Hematoma", "Hemorrhagic atheromatous plaque", "Impatent structure", "Induration", "Intramural hemorrhage", "Invagination", "Loose body", "Narrowed structure", "Perforation", "Petechia", "Pseudoaneurysm", "Purpura", "Separation", "Sequestration", "Sequestrum", "Sinus", "Splinter hemorrhage", "Subcapsular hemorrhage", "Submucosal hemorrhage", "Telangiectasis", "Ulcer", "Valvular insufficiency", "Wound hemorrhage"
}
# -----------------------------
# Define Set 2
# -----------------------------
set2 = {
    "Blood clot", "Ankylosis", "Hematoma", "Fistula", "False passage", "Sequestrum", "Entrapment", "Sequestration", "Intramural hemorrhage", "Wound hemorrhage", "Invagination", "Fenestration", "Dissecting hemorrhage", "Cyst", "Canalization", "Diverticulum", "Subcapsular hemorrhage", "Ecchymosis", "Bulging", "Acquired web", "Failure of fusion", "Fracture", "Submucosal hemorrhage", "Collapse", "Acquired absence", "Fissure", "Narrowed structure", "Sinus", "Pseudoaneurysm", "Purpura", "Ulcer", "Loose body", "Hemorrhagic atheromatous plaque", "Accumulation of fluid", "Fragmentation", "Disintegration", "Disruption", "Aneurysm", "Hematocele", "Valvular insufficiency", "Telangiectasis", "Cavity", "Impatent structure", "Induration", "Fibrin strand", "Perforation", "Calculus", "Separation", "Petechia", "Compression", "Splinter hemorrhage"
}

# -----------------------------
# Compute intersection
# -----------------------------
intersection = set1.intersection(set2)

set1 = set1 - set2

# -----------------------------
# Output
# -----------------------------
print("Set 1 size (after removal):", len(set1))
print("Set 2 size:", len(set2))
print("Intersection size:", len(intersection))

print("\nIntersection concepts:")
for item in sorted(intersection):
    print("-", item)
```