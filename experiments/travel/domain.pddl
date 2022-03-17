(define (domain travel)
  (:requirements :strips :typing)
  (:types car plane state)
  (:predicates (isredplane ?v - plane)
	(caravailable ?v - car)
	(planeavailable ?v - plane)
	(fly ?v - state ?vv - plane)
	(visited ?v - state)
	(adjacent ?v - state ?vv - state)
	(isblueplane ?v - plane)
	(isbluestate ?v - state)
	(isredstate ?v - state)
	(at ?v - state)
	(drive ?v - state ?vv - state ?vvv - car)
	(walk ?v - state)
  )


	(:action fly-red
		:parameters (?from - state ?to - state ?plane - plane)
		:precondition (and (at ?from)
			(isredstate ?from)
			(isredstate ?to)
			(isredplane ?plane)
			(planeavailable ?plane)
			(fly ?to ?plane))
		:effect (and
			(not (at ?from))
			(at ?to)
			(not (planeavailable ?plane))
			(visited ?to))
	)


	(:action drive
		:parameters (?from - state ?to - state ?thru - state ?car - car)
		:precondition (and (at ?from)
			(caravailable ?car)
			(adjacent ?from ?thru)
			(adjacent ?thru ?to)
			(drive ?to ?thru ?car))
		:effect (and
			(not (at ?from))
			(at ?to)
			(not (caravailable ?car))
			(visited ?to))
	)


	(:action fly-blue
		:parameters (?from - state ?to - state ?plane - plane)
		:precondition (and (at ?from)
			(isbluestate ?from)
			(isbluestate ?to)
			(isblueplane ?plane)
			(planeavailable ?plane)
			(fly ?to ?plane))
		:effect (and
			(not (at ?from))
			(at ?to)
			(not (planeavailable ?plane))
			(visited ?to))
	)


	(:action walk
		:parameters (?from - state ?to - state)
		:precondition (and (at ?from)
			(adjacent ?from ?to)
			(walk ?to))
		:effect (and
			(not (at ?from))
			(at ?to)
			(visited ?to))
	)

)
