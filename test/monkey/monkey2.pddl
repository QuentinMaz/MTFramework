(define (problem monkey2)
    (:domain monkey)
    (:objects l1 l2 l3 l4 l5 l6)
    (:init
        (location l1)
        (location l2)
        (location l3)
        (location l4)
        (location l5)
        (location l6)
        (at monkey l1)
        (onfloor)
        (at box l2)
        (at bananas l3)
        (at knife l4)
        (at waterfountain l5)
        (at glass l6)
    )
    (:goal
    (
        and
        (hasbananas)
        (haswater)
    ))
)
