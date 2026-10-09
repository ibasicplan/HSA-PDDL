(define (domain zenotravel)
  (:requirements :strips :typing)
  (:types person aircraft city flevel)
  (:predicates
    (at ?x - object ?c - city)
    (in ?p - person ?a - aircraft)
    (fuel-level ?a - aircraft ?l - flevel)
    (next ?l1 - flevel ?l2 - flevel)
  )
  (:action board
    :parameters (?p - person ?a - aircraft ?c - city)
    :precondition (and (at ?p ?c) (at ?a ?c))
    :effect (and (not (at ?p ?c)) (in ?p ?a))
  )
  (:action debark
    :parameters (?p - person ?a - aircraft ?c - city)
    :precondition (and (in ?p ?a) (at ?a ?c))
    :effect (and (not (in ?p ?a)) (at ?p ?c))
  )
)

