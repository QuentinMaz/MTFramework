
(define (problem travel) (:domain travel)
  (:objects
        ca - state
	car-0 - car
	ky - state
	nj - state
	og - state
	pe - state
	plane-0 - plane
	plane-1 - plane
	tn - state
	tx - state
	wv - state
  )

  (:init
	(drive ca ca car-0)
	(drive ca ky car-0)
	(drive ca nj car-0)
	(drive ca og car-0)
	(drive ca pe car-0)
	(drive ca tn car-0)
	(drive ca tx car-0)
	(drive ca wv car-0)
	(drive ky ca car-0)
	(drive ky ky car-0)
	(drive ky nj car-0)
	(drive ky og car-0)
	(drive ky pe car-0)
	(drive ky tn car-0)
	(drive ky tx car-0)
	(drive ky wv car-0)
	(drive nj ca car-0)
	(drive nj ky car-0)
	(drive nj nj car-0)
	(drive nj og car-0)
	(drive nj pe car-0)
	(drive nj tn car-0)
	(drive nj tx car-0)
	(drive nj wv car-0)
	(drive og ca car-0)
	(drive og ky car-0)
	(drive og nj car-0)
	(drive og og car-0)
	(drive og pe car-0)
	(drive og tn car-0)
	(drive og tx car-0)
	(drive og wv car-0)
	(drive pe ca car-0)
	(drive pe ky car-0)
	(drive pe nj car-0)
	(drive pe og car-0)
	(drive pe pe car-0)
	(drive pe tn car-0)
	(drive pe tx car-0)
	(drive pe wv car-0)
	(drive tn ca car-0)
	(drive tn ky car-0)
	(drive tn nj car-0)
	(drive tn og car-0)
	(drive tn pe car-0)
	(drive tn tn car-0)
	(drive tn tx car-0)
	(drive tn wv car-0)
	(drive tx ca car-0)
	(drive tx ky car-0)
	(drive tx nj car-0)
	(drive tx og car-0)
	(drive tx pe car-0)
	(drive tx tn car-0)
	(drive tx tx car-0)
	(drive tx wv car-0)
	(drive wv ca car-0)
	(drive wv ky car-0)
	(drive wv nj car-0)
	(drive wv og car-0)
	(drive wv pe car-0)
	(drive wv tn car-0)
	(drive wv tx car-0)
	(drive wv wv car-0)
	(fly ca plane-0)
	(fly ca plane-1)
	(fly ky plane-0)
	(fly ky plane-1)
	(fly nj plane-0)
	(fly nj plane-1)
	(fly og plane-0)
	(fly og plane-1)
	(fly pe plane-0)
	(fly pe plane-1)
	(fly tn plane-0)
	(fly tn plane-1)
	(fly tx plane-0)
	(fly tx plane-1)
	(fly wv plane-0)
	(fly wv plane-1)
	(walk ca)
	(walk ky)
	(walk nj)
	(walk og)
	(walk pe)
	(walk tn)
	(walk tx)
	(walk wv)
	(adjacent ca og)
	(adjacent ky tn)
	(adjacent ky wv)
	(adjacent nj pe)
	(adjacent og ca)
	(adjacent pe nj)
	(adjacent pe wv)
	(adjacent tn ky)
	(adjacent wv ky)
	(adjacent wv pe)
	(at nj)
	(caravailable car-0)
	(isblueplane plane-1)
	(isbluestate ca)
	(isbluestate tx)
	(isredplane plane-0)
	(isredstate ky)
	(isredstate og)
	(planeavailable plane-0)
	(planeavailable plane-1)
)
  (:goal (and
	(visited tx)
	(visited tn))))
