(define (domain monkey)
(:requirements :strips)
(:constants monkey box knife bananas glass waterfountain)
(:predicates
    (location ?x)
    (onfloor)
    (at ?m ?x)
    (hasknife)
    (onbox ?x)
    (hasbananas)
    (hasglass)
    (haswater)
)

;; goes from x to y
(:action GOTO
    :parameters (?x ?y)
    :precondition
    (
        and
        (location ?x)
        (location ?y)
        (onfloor)
        (at monkey ?x)
    )
    :effect
    (
        and
        (at monkey ?y)
        (not (at monkey ?x))
    )
)

;; climbs
(:action CLIMB
    :parameters (?x)
    :precondition
    (
        and
        (location ?x)
        (at box ?x)
        (at monkey ?x)
        (onfloor)
    )
    :effect
    (
        and
        (onbox ?x)
        (not (onfloor))
    )
)

;; descends from the box
(:action DESCEND
    :parameters (?x)
    :precondition
    (
        and
        (location ?x)
        (at box ?x)
        (at monkey ?x)
        (onbox ?x)
    )
    :effect
    (
        and
        (onfloor)
        (not (onbox ?x))
    )
)

;; pushes the box from x to y
(:action PUSHBOX
    :parameters (?x ?y)
    :precondition
    (
        and
        (location ?x)
        (location ?y)
        (at box ?x)
        (at monkey ?x)
        (onfloor)
    )
    :effect
    (
        and
        (at monkey ?y)
        (not (at monkey ?x))
        (at box ?y)
        (not (at box ?x))
    )
)

;; gets the knife
(:action GETKNIFE
    :parameters (?y)
    :precondition
    (
        and
        (location ?y)
        (at knife ?y)
        (at monkey ?y)
        (onfloor)
    )
    :effect
    (
        and
        (hasknife)
        (not (at knife ?y))
    )
)

;; gets bananas
(:action GRABBANANAS
    :parameters (?y)
    :precondition
    (
        and
        (location ?y)
        (hasknife)
        (at bananas ?y)
        (onbox ?y)
        (at monkey ?y)
    )
    :effect
        (hasbananas)
    )

;; gets glass
(:action PICKGLASS
    :parameters (?y)
    :precondition
    (
        and
        (location ?y)
        (at glass ?y)
        (at monkey ?y)
        (onfloor)
    )
    :effect
    (
        and
        (hasglass)
        (not (at glass ?y))
    )
)

;; gets water
(:action GETWATER
    :parameters (?y)
    :precondition
    (
        and
        (location ?y)
        (hasglass)
        (at waterfountain ?y)
        (at monkey ?y)
        (onbox ?y)
    )
    :effect
        (haswater)
    )
)