;;; ============================================================
;;; RECHTECKE UM LINIEN (RUL)
;;; Erzeugt Rechtecke um ausgewaehlte Linien.
;;; Breite wird vom Benutzer eingegeben,
;;; Laenge entspricht der Linienlaenge.
;;; ============================================================

(defun c:RUL (/ ss breite i ent ed p1 p2 dx dy laenge nx ny
                off1 off2 c1 c2 c3 c4 old-layer)

  ;; Layer "RECHTECKE" erstellen falls nicht vorhanden
  (if (not (tblsearch "LAYER" "RECHTECKE"))
    (entmake (list '(0 . "LAYER")
                   '(100 . "AcDbSymbolTableRecord")
                   '(100 . "AcDbLayerTableRecord")
                   '(2 . "RECHTECKE")
                   '(70 . 0)
                   '(62 . 3) ; gruen
            )
  )

  ;; Linien auswaehlen
  (princ "\nLinien auswaehlen fuer Rechteck-Erzeugung...")
  (setq ss (ssget '((0 . "LINE"))))

  (if ss
    (progn
      ;; Breite abfragen
      (setq breite (getreal "\nBreite der Rechtecke eingeben: "))

      (if (and breite (> breite 0.0))
        (progn
          (setq i 0)

          ;; Alle ausgewaehlten Linien durchlaufen
          (repeat (sslength ss)
            (setq ent (ssname ss i))
            (setq ed (entget ent))

            ;; Start- und Endpunkt der Linie
            (setq p1 (cdr (assoc 10 ed)))  ; Startpunkt
            (setq p2 (cdr (assoc 11 ed)))  ; Endpunkt

            ;; Richtungsvektor berechnen
            (setq dx (- (car p2) (car p1)))
            (setq dy (- (cadr p2) (cadr p1)))
            (setq laenge (sqrt (+ (* dx dx) (* dy dy))))

            (if (> laenge 0.0)
              (progn
                ;; Normalvektor (senkrecht zur Linie), normiert
                (setq nx (/ (- dy) laenge))
                (setq ny (/ dx laenge))

                ;; Offset = halbe Breite
                (setq off1 (* nx (/ breite 2.0)))
                (setq off2 (* ny (/ breite 2.0)))

                ;; 4 Eckpunkte des Rechtecks
                (setq c1 (list (+ (car p1) off1) (+ (cadr p1) off2) 0.0))
                (setq c2 (list (- (car p1) off1) (- (cadr p1) off2) 0.0))
                (setq c3 (list (- (car p2) off1) (- (cadr p2) off2) 0.0))
                (setq c4 (list (+ (car p2) off1) (+ (cadr p2) off2) 0.0))

                ;; Geschlossene LWPOLYLINE erzeugen
                (entmake
                  (list '(0 . "LWPOLYLINE")
                        '(100 . "AcDbEntity")
                        '(8 . "RECHTECKE")    ; Layer
                        '(100 . "AcDbPolyline")
                        '(90 . 4)              ; 4 Eckpunkte
                        '(70 . 1)              ; geschlossen
                        (cons 10 (list (car c1) (cadr c1)))
                        (cons 10 (list (car c4) (cadr c4)))
                        (cons 10 (list (car c3) (cadr c3)))
                        (cons 10 (list (car c2) (cadr c2)))
                  )
                )
              )
            )

            (setq i (1+ i))
          ) ; repeat

          (princ (strcat "\n"
                         (itoa (sslength ss))
                         " Rechteck(e) erzeugt auf Layer RECHTECKE."))
        )
        (princ "\nUngueltige Breite. Abbruch.")
      )
    )
    (princ "\nKeine Linien ausgewaehlt. Abbruch.")
  )

  (princ)
)

;;; Lademeldung
(princ "\nBefehl RUL geladen. Eingabe: RUL")
(princ)
