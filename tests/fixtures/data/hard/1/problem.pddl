(define (problem zeno-fixture-hard-1)
  (:domain zenotravel)
  (:objects
    plane2 - aircraft
    person2 - person
    city2 city3 - city
    fl1 fl2 - flevel
  )
  (:init
    (at plane2 city2)
    (in person2 plane2)
    (fuel-level plane2 fl2)
    (next fl1 fl2)
  )
  (:goal
    (and
      (at person2 city3)
    )
  )
)

