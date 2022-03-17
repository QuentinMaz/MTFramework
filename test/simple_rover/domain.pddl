(define (domain simple_rover)
(:requirements :strips :typing)
(:types location data)
(:predicates
 (at ?x - location)
 (avail ?d - data ?x - location)
 (comm ?d - data)
 (have ?d - data))
(:action drive
 :parameters (?x - location ?y - location)
 :precondition (at ?x)
 :effect (and (at ?y) (not (at ?x))))
(:action commun
 :parameters (?d - data)
 :precondition (have ?d)
 :effect (comm ?d))
(:action sample
 :parameters (?d - data ?x - location)
 :precondition (and (at ?x) (avail ?d ?x))
 :effect (have ?d))
)