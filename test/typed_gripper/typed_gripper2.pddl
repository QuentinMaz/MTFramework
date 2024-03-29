(define (problem typed_gripper4)
    (:domain gripper-strips)
    (:objects
        rooma roomb - room
        left right - gripper
        ball1 ball2 ball3 ball4 - ball)
    (:init
        (free left) (free right)
        (at ball1 rooma) (at ball2 rooma) (at ball3 rooma) (at ball4 rooma)
        (at-robby rooma)
    )
    (:goal 
        (and 
            (at ball1 roomb)
            (at ball2 roomb)
            (at ball3 roomb)
            (at ball4 roomb)
        )
    )
)