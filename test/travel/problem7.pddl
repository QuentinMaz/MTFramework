
(define (problem travel) (:domain travel)
  (:objects
        car-0 - car
	car-1 - car
	car-2 - car
	car-3 - car
	fl - state
	hi - state
	ia - state
	id - state
	il - state
	mi - state
	mn - state
	mo - state
	nd - state
	ok - state
	plane-0 - plane
	plane-1 - plane
	plane-2 - plane
	plane-3 - plane
	sd - state
	tx - state
	wa - state
	wn - state
  )

  (:init
	(drive fl fl car-0)
	(drive fl fl car-1)
	(drive fl fl car-2)
	(drive fl fl car-3)
	(drive fl hi car-0)
	(drive fl hi car-1)
	(drive fl hi car-2)
	(drive fl hi car-3)
	(drive fl ia car-0)
	(drive fl ia car-1)
	(drive fl ia car-2)
	(drive fl ia car-3)
	(drive fl id car-0)
	(drive fl id car-1)
	(drive fl id car-2)
	(drive fl id car-3)
	(drive fl il car-0)
	(drive fl il car-1)
	(drive fl il car-2)
	(drive fl il car-3)
	(drive fl mi car-0)
	(drive fl mi car-1)
	(drive fl mi car-2)
	(drive fl mi car-3)
	(drive fl mn car-0)
	(drive fl mn car-1)
	(drive fl mn car-2)
	(drive fl mn car-3)
	(drive fl mo car-0)
	(drive fl mo car-1)
	(drive fl mo car-2)
	(drive fl mo car-3)
	(drive fl nd car-0)
	(drive fl nd car-1)
	(drive fl nd car-2)
	(drive fl nd car-3)
	(drive fl ok car-0)
	(drive fl ok car-1)
	(drive fl ok car-2)
	(drive fl ok car-3)
	(drive fl sd car-0)
	(drive fl sd car-1)
	(drive fl sd car-2)
	(drive fl sd car-3)
	(drive fl tx car-0)
	(drive fl tx car-1)
	(drive fl tx car-2)
	(drive fl tx car-3)
	(drive fl wa car-0)
	(drive fl wa car-1)
	(drive fl wa car-2)
	(drive fl wa car-3)
	(drive fl wn car-0)
	(drive fl wn car-1)
	(drive fl wn car-2)
	(drive fl wn car-3)
	(drive hi fl car-0)
	(drive hi fl car-1)
	(drive hi fl car-2)
	(drive hi fl car-3)
	(drive hi hi car-0)
	(drive hi hi car-1)
	(drive hi hi car-2)
	(drive hi hi car-3)
	(drive hi ia car-0)
	(drive hi ia car-1)
	(drive hi ia car-2)
	(drive hi ia car-3)
	(drive hi id car-0)
	(drive hi id car-1)
	(drive hi id car-2)
	(drive hi id car-3)
	(drive hi il car-0)
	(drive hi il car-1)
	(drive hi il car-2)
	(drive hi il car-3)
	(drive hi mi car-0)
	(drive hi mi car-1)
	(drive hi mi car-2)
	(drive hi mi car-3)
	(drive hi mn car-0)
	(drive hi mn car-1)
	(drive hi mn car-2)
	(drive hi mn car-3)
	(drive hi mo car-0)
	(drive hi mo car-1)
	(drive hi mo car-2)
	(drive hi mo car-3)
	(drive hi nd car-0)
	(drive hi nd car-1)
	(drive hi nd car-2)
	(drive hi nd car-3)
	(drive hi ok car-0)
	(drive hi ok car-1)
	(drive hi ok car-2)
	(drive hi ok car-3)
	(drive hi sd car-0)
	(drive hi sd car-1)
	(drive hi sd car-2)
	(drive hi sd car-3)
	(drive hi tx car-0)
	(drive hi tx car-1)
	(drive hi tx car-2)
	(drive hi tx car-3)
	(drive hi wa car-0)
	(drive hi wa car-1)
	(drive hi wa car-2)
	(drive hi wa car-3)
	(drive hi wn car-0)
	(drive hi wn car-1)
	(drive hi wn car-2)
	(drive hi wn car-3)
	(drive ia fl car-0)
	(drive ia fl car-1)
	(drive ia fl car-2)
	(drive ia fl car-3)
	(drive ia hi car-0)
	(drive ia hi car-1)
	(drive ia hi car-2)
	(drive ia hi car-3)
	(drive ia ia car-0)
	(drive ia ia car-1)
	(drive ia ia car-2)
	(drive ia ia car-3)
	(drive ia id car-0)
	(drive ia id car-1)
	(drive ia id car-2)
	(drive ia id car-3)
	(drive ia il car-0)
	(drive ia il car-1)
	(drive ia il car-2)
	(drive ia il car-3)
	(drive ia mi car-0)
	(drive ia mi car-1)
	(drive ia mi car-2)
	(drive ia mi car-3)
	(drive ia mn car-0)
	(drive ia mn car-1)
	(drive ia mn car-2)
	(drive ia mn car-3)
	(drive ia mo car-0)
	(drive ia mo car-1)
	(drive ia mo car-2)
	(drive ia mo car-3)
	(drive ia nd car-0)
	(drive ia nd car-1)
	(drive ia nd car-2)
	(drive ia nd car-3)
	(drive ia ok car-0)
	(drive ia ok car-1)
	(drive ia ok car-2)
	(drive ia ok car-3)
	(drive ia sd car-0)
	(drive ia sd car-1)
	(drive ia sd car-2)
	(drive ia sd car-3)
	(drive ia tx car-0)
	(drive ia tx car-1)
	(drive ia tx car-2)
	(drive ia tx car-3)
	(drive ia wa car-0)
	(drive ia wa car-1)
	(drive ia wa car-2)
	(drive ia wa car-3)
	(drive ia wn car-0)
	(drive ia wn car-1)
	(drive ia wn car-2)
	(drive ia wn car-3)
	(drive id fl car-0)
	(drive id fl car-1)
	(drive id fl car-2)
	(drive id fl car-3)
	(drive id hi car-0)
	(drive id hi car-1)
	(drive id hi car-2)
	(drive id hi car-3)
	(drive id ia car-0)
	(drive id ia car-1)
	(drive id ia car-2)
	(drive id ia car-3)
	(drive id id car-0)
	(drive id id car-1)
	(drive id id car-2)
	(drive id id car-3)
	(drive id il car-0)
	(drive id il car-1)
	(drive id il car-2)
	(drive id il car-3)
	(drive id mi car-0)
	(drive id mi car-1)
	(drive id mi car-2)
	(drive id mi car-3)
	(drive id mn car-0)
	(drive id mn car-1)
	(drive id mn car-2)
	(drive id mn car-3)
	(drive id mo car-0)
	(drive id mo car-1)
	(drive id mo car-2)
	(drive id mo car-3)
	(drive id nd car-0)
	(drive id nd car-1)
	(drive id nd car-2)
	(drive id nd car-3)
	(drive id ok car-0)
	(drive id ok car-1)
	(drive id ok car-2)
	(drive id ok car-3)
	(drive id sd car-0)
	(drive id sd car-1)
	(drive id sd car-2)
	(drive id sd car-3)
	(drive id tx car-0)
	(drive id tx car-1)
	(drive id tx car-2)
	(drive id tx car-3)
	(drive id wa car-0)
	(drive id wa car-1)
	(drive id wa car-2)
	(drive id wa car-3)
	(drive id wn car-0)
	(drive id wn car-1)
	(drive id wn car-2)
	(drive id wn car-3)
	(drive il fl car-0)
	(drive il fl car-1)
	(drive il fl car-2)
	(drive il fl car-3)
	(drive il hi car-0)
	(drive il hi car-1)
	(drive il hi car-2)
	(drive il hi car-3)
	(drive il ia car-0)
	(drive il ia car-1)
	(drive il ia car-2)
	(drive il ia car-3)
	(drive il id car-0)
	(drive il id car-1)
	(drive il id car-2)
	(drive il id car-3)
	(drive il il car-0)
	(drive il il car-1)
	(drive il il car-2)
	(drive il il car-3)
	(drive il mi car-0)
	(drive il mi car-1)
	(drive il mi car-2)
	(drive il mi car-3)
	(drive il mn car-0)
	(drive il mn car-1)
	(drive il mn car-2)
	(drive il mn car-3)
	(drive il mo car-0)
	(drive il mo car-1)
	(drive il mo car-2)
	(drive il mo car-3)
	(drive il nd car-0)
	(drive il nd car-1)
	(drive il nd car-2)
	(drive il nd car-3)
	(drive il ok car-0)
	(drive il ok car-1)
	(drive il ok car-2)
	(drive il ok car-3)
	(drive il sd car-0)
	(drive il sd car-1)
	(drive il sd car-2)
	(drive il sd car-3)
	(drive il tx car-0)
	(drive il tx car-1)
	(drive il tx car-2)
	(drive il tx car-3)
	(drive il wa car-0)
	(drive il wa car-1)
	(drive il wa car-2)
	(drive il wa car-3)
	(drive il wn car-0)
	(drive il wn car-1)
	(drive il wn car-2)
	(drive il wn car-3)
	(drive mi fl car-0)
	(drive mi fl car-1)
	(drive mi fl car-2)
	(drive mi fl car-3)
	(drive mi hi car-0)
	(drive mi hi car-1)
	(drive mi hi car-2)
	(drive mi hi car-3)
	(drive mi ia car-0)
	(drive mi ia car-1)
	(drive mi ia car-2)
	(drive mi ia car-3)
	(drive mi id car-0)
	(drive mi id car-1)
	(drive mi id car-2)
	(drive mi id car-3)
	(drive mi il car-0)
	(drive mi il car-1)
	(drive mi il car-2)
	(drive mi il car-3)
	(drive mi mi car-0)
	(drive mi mi car-1)
	(drive mi mi car-2)
	(drive mi mi car-3)
	(drive mi mn car-0)
	(drive mi mn car-1)
	(drive mi mn car-2)
	(drive mi mn car-3)
	(drive mi mo car-0)
	(drive mi mo car-1)
	(drive mi mo car-2)
	(drive mi mo car-3)
	(drive mi nd car-0)
	(drive mi nd car-1)
	(drive mi nd car-2)
	(drive mi nd car-3)
	(drive mi ok car-0)
	(drive mi ok car-1)
	(drive mi ok car-2)
	(drive mi ok car-3)
	(drive mi sd car-0)
	(drive mi sd car-1)
	(drive mi sd car-2)
	(drive mi sd car-3)
	(drive mi tx car-0)
	(drive mi tx car-1)
	(drive mi tx car-2)
	(drive mi tx car-3)
	(drive mi wa car-0)
	(drive mi wa car-1)
	(drive mi wa car-2)
	(drive mi wa car-3)
	(drive mi wn car-0)
	(drive mi wn car-1)
	(drive mi wn car-2)
	(drive mi wn car-3)
	(drive mn fl car-0)
	(drive mn fl car-1)
	(drive mn fl car-2)
	(drive mn fl car-3)
	(drive mn hi car-0)
	(drive mn hi car-1)
	(drive mn hi car-2)
	(drive mn hi car-3)
	(drive mn ia car-0)
	(drive mn ia car-1)
	(drive mn ia car-2)
	(drive mn ia car-3)
	(drive mn id car-0)
	(drive mn id car-1)
	(drive mn id car-2)
	(drive mn id car-3)
	(drive mn il car-0)
	(drive mn il car-1)
	(drive mn il car-2)
	(drive mn il car-3)
	(drive mn mi car-0)
	(drive mn mi car-1)
	(drive mn mi car-2)
	(drive mn mi car-3)
	(drive mn mn car-0)
	(drive mn mn car-1)
	(drive mn mn car-2)
	(drive mn mn car-3)
	(drive mn mo car-0)
	(drive mn mo car-1)
	(drive mn mo car-2)
	(drive mn mo car-3)
	(drive mn nd car-0)
	(drive mn nd car-1)
	(drive mn nd car-2)
	(drive mn nd car-3)
	(drive mn ok car-0)
	(drive mn ok car-1)
	(drive mn ok car-2)
	(drive mn ok car-3)
	(drive mn sd car-0)
	(drive mn sd car-1)
	(drive mn sd car-2)
	(drive mn sd car-3)
	(drive mn tx car-0)
	(drive mn tx car-1)
	(drive mn tx car-2)
	(drive mn tx car-3)
	(drive mn wa car-0)
	(drive mn wa car-1)
	(drive mn wa car-2)
	(drive mn wa car-3)
	(drive mn wn car-0)
	(drive mn wn car-1)
	(drive mn wn car-2)
	(drive mn wn car-3)
	(drive mo fl car-0)
	(drive mo fl car-1)
	(drive mo fl car-2)
	(drive mo fl car-3)
	(drive mo hi car-0)
	(drive mo hi car-1)
	(drive mo hi car-2)
	(drive mo hi car-3)
	(drive mo ia car-0)
	(drive mo ia car-1)
	(drive mo ia car-2)
	(drive mo ia car-3)
	(drive mo id car-0)
	(drive mo id car-1)
	(drive mo id car-2)
	(drive mo id car-3)
	(drive mo il car-0)
	(drive mo il car-1)
	(drive mo il car-2)
	(drive mo il car-3)
	(drive mo mi car-0)
	(drive mo mi car-1)
	(drive mo mi car-2)
	(drive mo mi car-3)
	(drive mo mn car-0)
	(drive mo mn car-1)
	(drive mo mn car-2)
	(drive mo mn car-3)
	(drive mo mo car-0)
	(drive mo mo car-1)
	(drive mo mo car-2)
	(drive mo mo car-3)
	(drive mo nd car-0)
	(drive mo nd car-1)
	(drive mo nd car-2)
	(drive mo nd car-3)
	(drive mo ok car-0)
	(drive mo ok car-1)
	(drive mo ok car-2)
	(drive mo ok car-3)
	(drive mo sd car-0)
	(drive mo sd car-1)
	(drive mo sd car-2)
	(drive mo sd car-3)
	(drive mo tx car-0)
	(drive mo tx car-1)
	(drive mo tx car-2)
	(drive mo tx car-3)
	(drive mo wa car-0)
	(drive mo wa car-1)
	(drive mo wa car-2)
	(drive mo wa car-3)
	(drive mo wn car-0)
	(drive mo wn car-1)
	(drive mo wn car-2)
	(drive mo wn car-3)
	(drive nd fl car-0)
	(drive nd fl car-1)
	(drive nd fl car-2)
	(drive nd fl car-3)
	(drive nd hi car-0)
	(drive nd hi car-1)
	(drive nd hi car-2)
	(drive nd hi car-3)
	(drive nd ia car-0)
	(drive nd ia car-1)
	(drive nd ia car-2)
	(drive nd ia car-3)
	(drive nd id car-0)
	(drive nd id car-1)
	(drive nd id car-2)
	(drive nd id car-3)
	(drive nd il car-0)
	(drive nd il car-1)
	(drive nd il car-2)
	(drive nd il car-3)
	(drive nd mi car-0)
	(drive nd mi car-1)
	(drive nd mi car-2)
	(drive nd mi car-3)
	(drive nd mn car-0)
	(drive nd mn car-1)
	(drive nd mn car-2)
	(drive nd mn car-3)
	(drive nd mo car-0)
	(drive nd mo car-1)
	(drive nd mo car-2)
	(drive nd mo car-3)
	(drive nd nd car-0)
	(drive nd nd car-1)
	(drive nd nd car-2)
	(drive nd nd car-3)
	(drive nd ok car-0)
	(drive nd ok car-1)
	(drive nd ok car-2)
	(drive nd ok car-3)
	(drive nd sd car-0)
	(drive nd sd car-1)
	(drive nd sd car-2)
	(drive nd sd car-3)
	(drive nd tx car-0)
	(drive nd tx car-1)
	(drive nd tx car-2)
	(drive nd tx car-3)
	(drive nd wa car-0)
	(drive nd wa car-1)
	(drive nd wa car-2)
	(drive nd wa car-3)
	(drive nd wn car-0)
	(drive nd wn car-1)
	(drive nd wn car-2)
	(drive nd wn car-3)
	(drive ok fl car-0)
	(drive ok fl car-1)
	(drive ok fl car-2)
	(drive ok fl car-3)
	(drive ok hi car-0)
	(drive ok hi car-1)
	(drive ok hi car-2)
	(drive ok hi car-3)
	(drive ok ia car-0)
	(drive ok ia car-1)
	(drive ok ia car-2)
	(drive ok ia car-3)
	(drive ok id car-0)
	(drive ok id car-1)
	(drive ok id car-2)
	(drive ok id car-3)
	(drive ok il car-0)
	(drive ok il car-1)
	(drive ok il car-2)
	(drive ok il car-3)
	(drive ok mi car-0)
	(drive ok mi car-1)
	(drive ok mi car-2)
	(drive ok mi car-3)
	(drive ok mn car-0)
	(drive ok mn car-1)
	(drive ok mn car-2)
	(drive ok mn car-3)
	(drive ok mo car-0)
	(drive ok mo car-1)
	(drive ok mo car-2)
	(drive ok mo car-3)
	(drive ok nd car-0)
	(drive ok nd car-1)
	(drive ok nd car-2)
	(drive ok nd car-3)
	(drive ok ok car-0)
	(drive ok ok car-1)
	(drive ok ok car-2)
	(drive ok ok car-3)
	(drive ok sd car-0)
	(drive ok sd car-1)
	(drive ok sd car-2)
	(drive ok sd car-3)
	(drive ok tx car-0)
	(drive ok tx car-1)
	(drive ok tx car-2)
	(drive ok tx car-3)
	(drive ok wa car-0)
	(drive ok wa car-1)
	(drive ok wa car-2)
	(drive ok wa car-3)
	(drive ok wn car-0)
	(drive ok wn car-1)
	(drive ok wn car-2)
	(drive ok wn car-3)
	(drive sd fl car-0)
	(drive sd fl car-1)
	(drive sd fl car-2)
	(drive sd fl car-3)
	(drive sd hi car-0)
	(drive sd hi car-1)
	(drive sd hi car-2)
	(drive sd hi car-3)
	(drive sd ia car-0)
	(drive sd ia car-1)
	(drive sd ia car-2)
	(drive sd ia car-3)
	(drive sd id car-0)
	(drive sd id car-1)
	(drive sd id car-2)
	(drive sd id car-3)
	(drive sd il car-0)
	(drive sd il car-1)
	(drive sd il car-2)
	(drive sd il car-3)
	(drive sd mi car-0)
	(drive sd mi car-1)
	(drive sd mi car-2)
	(drive sd mi car-3)
	(drive sd mn car-0)
	(drive sd mn car-1)
	(drive sd mn car-2)
	(drive sd mn car-3)
	(drive sd mo car-0)
	(drive sd mo car-1)
	(drive sd mo car-2)
	(drive sd mo car-3)
	(drive sd nd car-0)
	(drive sd nd car-1)
	(drive sd nd car-2)
	(drive sd nd car-3)
	(drive sd ok car-0)
	(drive sd ok car-1)
	(drive sd ok car-2)
	(drive sd ok car-3)
	(drive sd sd car-0)
	(drive sd sd car-1)
	(drive sd sd car-2)
	(drive sd sd car-3)
	(drive sd tx car-0)
	(drive sd tx car-1)
	(drive sd tx car-2)
	(drive sd tx car-3)
	(drive sd wa car-0)
	(drive sd wa car-1)
	(drive sd wa car-2)
	(drive sd wa car-3)
	(drive sd wn car-0)
	(drive sd wn car-1)
	(drive sd wn car-2)
	(drive sd wn car-3)
	(drive tx fl car-0)
	(drive tx fl car-1)
	(drive tx fl car-2)
	(drive tx fl car-3)
	(drive tx hi car-0)
	(drive tx hi car-1)
	(drive tx hi car-2)
	(drive tx hi car-3)
	(drive tx ia car-0)
	(drive tx ia car-1)
	(drive tx ia car-2)
	(drive tx ia car-3)
	(drive tx id car-0)
	(drive tx id car-1)
	(drive tx id car-2)
	(drive tx id car-3)
	(drive tx il car-0)
	(drive tx il car-1)
	(drive tx il car-2)
	(drive tx il car-3)
	(drive tx mi car-0)
	(drive tx mi car-1)
	(drive tx mi car-2)
	(drive tx mi car-3)
	(drive tx mn car-0)
	(drive tx mn car-1)
	(drive tx mn car-2)
	(drive tx mn car-3)
	(drive tx mo car-0)
	(drive tx mo car-1)
	(drive tx mo car-2)
	(drive tx mo car-3)
	(drive tx nd car-0)
	(drive tx nd car-1)
	(drive tx nd car-2)
	(drive tx nd car-3)
	(drive tx ok car-0)
	(drive tx ok car-1)
	(drive tx ok car-2)
	(drive tx ok car-3)
	(drive tx sd car-0)
	(drive tx sd car-1)
	(drive tx sd car-2)
	(drive tx sd car-3)
	(drive tx tx car-0)
	(drive tx tx car-1)
	(drive tx tx car-2)
	(drive tx tx car-3)
	(drive tx wa car-0)
	(drive tx wa car-1)
	(drive tx wa car-2)
	(drive tx wa car-3)
	(drive tx wn car-0)
	(drive tx wn car-1)
	(drive tx wn car-2)
	(drive tx wn car-3)
	(drive wa fl car-0)
	(drive wa fl car-1)
	(drive wa fl car-2)
	(drive wa fl car-3)
	(drive wa hi car-0)
	(drive wa hi car-1)
	(drive wa hi car-2)
	(drive wa hi car-3)
	(drive wa ia car-0)
	(drive wa ia car-1)
	(drive wa ia car-2)
	(drive wa ia car-3)
	(drive wa id car-0)
	(drive wa id car-1)
	(drive wa id car-2)
	(drive wa id car-3)
	(drive wa il car-0)
	(drive wa il car-1)
	(drive wa il car-2)
	(drive wa il car-3)
	(drive wa mi car-0)
	(drive wa mi car-1)
	(drive wa mi car-2)
	(drive wa mi car-3)
	(drive wa mn car-0)
	(drive wa mn car-1)
	(drive wa mn car-2)
	(drive wa mn car-3)
	(drive wa mo car-0)
	(drive wa mo car-1)
	(drive wa mo car-2)
	(drive wa mo car-3)
	(drive wa nd car-0)
	(drive wa nd car-1)
	(drive wa nd car-2)
	(drive wa nd car-3)
	(drive wa ok car-0)
	(drive wa ok car-1)
	(drive wa ok car-2)
	(drive wa ok car-3)
	(drive wa sd car-0)
	(drive wa sd car-1)
	(drive wa sd car-2)
	(drive wa sd car-3)
	(drive wa tx car-0)
	(drive wa tx car-1)
	(drive wa tx car-2)
	(drive wa tx car-3)
	(drive wa wa car-0)
	(drive wa wa car-1)
	(drive wa wa car-2)
	(drive wa wa car-3)
	(drive wa wn car-0)
	(drive wa wn car-1)
	(drive wa wn car-2)
	(drive wa wn car-3)
	(drive wn fl car-0)
	(drive wn fl car-1)
	(drive wn fl car-2)
	(drive wn fl car-3)
	(drive wn hi car-0)
	(drive wn hi car-1)
	(drive wn hi car-2)
	(drive wn hi car-3)
	(drive wn ia car-0)
	(drive wn ia car-1)
	(drive wn ia car-2)
	(drive wn ia car-3)
	(drive wn id car-0)
	(drive wn id car-1)
	(drive wn id car-2)
	(drive wn id car-3)
	(drive wn il car-0)
	(drive wn il car-1)
	(drive wn il car-2)
	(drive wn il car-3)
	(drive wn mi car-0)
	(drive wn mi car-1)
	(drive wn mi car-2)
	(drive wn mi car-3)
	(drive wn mn car-0)
	(drive wn mn car-1)
	(drive wn mn car-2)
	(drive wn mn car-3)
	(drive wn mo car-0)
	(drive wn mo car-1)
	(drive wn mo car-2)
	(drive wn mo car-3)
	(drive wn nd car-0)
	(drive wn nd car-1)
	(drive wn nd car-2)
	(drive wn nd car-3)
	(drive wn ok car-0)
	(drive wn ok car-1)
	(drive wn ok car-2)
	(drive wn ok car-3)
	(drive wn sd car-0)
	(drive wn sd car-1)
	(drive wn sd car-2)
	(drive wn sd car-3)
	(drive wn tx car-0)
	(drive wn tx car-1)
	(drive wn tx car-2)
	(drive wn tx car-3)
	(drive wn wa car-0)
	(drive wn wa car-1)
	(drive wn wa car-2)
	(drive wn wa car-3)
	(drive wn wn car-0)
	(drive wn wn car-1)
	(drive wn wn car-2)
	(drive wn wn car-3)
	(fly fl plane-0)
	(fly fl plane-1)
	(fly fl plane-2)
	(fly fl plane-3)
	(fly hi plane-0)
	(fly hi plane-1)
	(fly hi plane-2)
	(fly hi plane-3)
	(fly ia plane-0)
	(fly ia plane-1)
	(fly ia plane-2)
	(fly ia plane-3)
	(fly id plane-0)
	(fly id plane-1)
	(fly id plane-2)
	(fly id plane-3)
	(fly il plane-0)
	(fly il plane-1)
	(fly il plane-2)
	(fly il plane-3)
	(fly mi plane-0)
	(fly mi plane-1)
	(fly mi plane-2)
	(fly mi plane-3)
	(fly mn plane-0)
	(fly mn plane-1)
	(fly mn plane-2)
	(fly mn plane-3)
	(fly mo plane-0)
	(fly mo plane-1)
	(fly mo plane-2)
	(fly mo plane-3)
	(fly nd plane-0)
	(fly nd plane-1)
	(fly nd plane-2)
	(fly nd plane-3)
	(fly ok plane-0)
	(fly ok plane-1)
	(fly ok plane-2)
	(fly ok plane-3)
	(fly sd plane-0)
	(fly sd plane-1)
	(fly sd plane-2)
	(fly sd plane-3)
	(fly tx plane-0)
	(fly tx plane-1)
	(fly tx plane-2)
	(fly tx plane-3)
	(fly wa plane-0)
	(fly wa plane-1)
	(fly wa plane-2)
	(fly wa plane-3)
	(fly wn plane-0)
	(fly wn plane-1)
	(fly wn plane-2)
	(fly wn plane-3)
	(walk fl)
	(walk hi)
	(walk ia)
	(walk id)
	(walk il)
	(walk mi)
	(walk mn)
	(walk mo)
	(walk nd)
	(walk ok)
	(walk sd)
	(walk tx)
	(walk wa)
	(walk wn)
	(adjacent ia il)
	(adjacent ia mn)
	(adjacent ia sd)
	(adjacent ia wn)
	(adjacent id mo)
	(adjacent id wa)
	(adjacent il ia)
	(adjacent mi wn)
	(adjacent mn ia)
	(adjacent mn nd)
	(adjacent mn sd)
	(adjacent mn wn)
	(adjacent mo id)
	(adjacent mo nd)
	(adjacent mo sd)
	(adjacent nd mn)
	(adjacent nd mo)
	(adjacent nd sd)
	(adjacent ok tx)
	(adjacent sd ia)
	(adjacent sd mn)
	(adjacent sd mo)
	(adjacent sd nd)
	(adjacent tx ok)
	(adjacent wa id)
	(adjacent wn ia)
	(adjacent wn mi)
	(adjacent wn mn)
	(at sd)
	(caravailable car-0)
	(caravailable car-1)
	(caravailable car-2)
	(caravailable car-3)
	(isblueplane plane-2)
	(isblueplane plane-3)
	(isbluestate fl)
	(isbluestate ok)
	(isbluestate wa)
	(isbluestate hi)
    (isblueplane plane-0)
    (isblueplane plane-1)
	(planeavailable plane-0)
	(planeavailable plane-1)
	(planeavailable plane-2)
	(planeavailable plane-3)
)  (:goal (and
	(visited wa)
	(visited fl)
	(visited ok)
	(visited id))))
