(define (problem complex_rover1)

    (:domain
        complex_rover
    )

    (:objects
        waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6

        sample1 sample2 sample3

        objective1 objective2

        rover1
    )

    (:init

        (waypoint waypoint1) (waypoint waypoint2) (waypoint waypoint3)
        (waypoint waypoint4) (waypoint waypoint5) (waypoint waypoint6)

        (sample sample1) (sample sample2) (sample sample3)

        (objective objective1) (objective objective2)

        (can-move waypoint1 waypoint5) (can-move waypoint1 waypoint4)
        (can-move waypoint2 waypoint5)
        (can-move waypoint3 waypoint4) (can-move waypoint3 waypoint6) (can-move waypoint3 waypoint5)
        (can-move waypoint4 waypoint3)
        (can-move waypoint5 waypoint1) (can-move waypoint5 waypoint2)
        (can-move waypoint6 waypoint3) (can-move waypoint6 waypoint2)

        (is-visible objective1 waypoint2) (is-visible objective1 waypoint6)
        (is-visible objective2 waypoint3) (is-visible objective2 waypoint4)

        (is-in sample1 waypoint5)
        (is-in sample2 waypoint3)
        (is-in sample3 waypoint4)

        (is-dropping-dock waypoint1)

        (rover rover1)
        (empty rover1)
        (at rover1 waypoint1)
    )

    (:goal
        (and
            (stored-sample sample1)
            (stored-sample sample2)
            (stored-sample sample3)

            (taken-image objective1)
            (taken-image objective2)

            (at rover1 waypoint1))
    )
)