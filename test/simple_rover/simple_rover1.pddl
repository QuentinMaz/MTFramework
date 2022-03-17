(define (problem simple_rover1)
 (:domain simple_rover)
 (:objects
 soil image rock - data
 alpha beta gamma - location)
 (:init (at alpha)
 (avail soil alpha)
 (avail rock beta)
 (avail image gamma))
 (:goal (and (comm soil)
 (comm image)
 (comm rock)))
)