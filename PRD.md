
# 📄 PRD — Portal 4 “Capstones, Benchmarks y Carrera Profesional”

## 1. Introducción

### 1.1 Propósito

El **Portal 4** es la culminación del programa. Aquí los alumnos aplican todo lo aprendido en Portales 1–3 para **diseñar, construir y desplegar proyectos de IA completos**, con benchmarks, documentación profesional y orientación laboral.
La meta es que cada alumno salga con un **portfolio demostrable** y la **preparación necesaria para entrevistas técnicas y roles de AI Engineer / LLMOps**.

### 1.2 Alcance

Este portal cubre:

* **Capstones de alto impacto** (elegibles en distintos dominios: dev-tools, RAG empresarial, copilotos, analítica).
* **Benchmarks de rendimiento y coste** en proyectos reales.
* **Documentación y pitches técnicos** estilo entrevista.
* **Simulacros de entrevistas técnicas y de diseño de sistemas**.
* **Guía de mercado laboral y negociación salarial**.

No incluye: fundamentos o prácticas básicas (son prerrequisito de Portales 1–3).

---

## 2. Público objetivo y usuarios

### 2.1 Perfil primario

* Devs que completaron Portales 1–3 o con experiencia equivalente en agentes/RAG/LLMOps.
* Perfil: **mid/senior** aspirando a **AI Engineer, Applied AI Dev, AI Ops Engineer, Product AI Lead**.

### 2.2 Problemas a resolver

* Falta de proyectos completos para mostrar en portfolio.
* Inseguridad al enfrentar entrevistas técnicas.
* Desconocimiento del mercado laboral actual en IA.
* Falta de estrategia para presentarse a empresas de alto nivel.

---

## 3. Objetivos y métricas de éxito

### 3.1 Objetivos de producto

1. Que cada alumno construya al menos **1 capstone robusto** con agentes, RAG híbrido, seguridad y métricas.
2. Generar un **benchmark comparativo** (latencia, precisión, coste) en su proyecto.
3. Crear un **pitch técnico** estilo entrevista (10 min).
4. Practicar en **simulacros de entrevistas** con feedback.
5. Orientar sobre **mercado laboral, roles y negociación**.

### 3.2 KPIs

* % de alumnos que completan al menos 1 capstone ≥ 70%.
* % que produce documentación lista para portfolio ≥ 60%.
* % que aprueban simulacros de entrevista ≥ 50%.
* NPS del portal ≥ 9/10.

---

## 4. Requisitos funcionales

### 4.1 Currículo

* **Módulo A — Selección de capstone**
  *Opciones: copiloto dev, asistente empresarial, copiloto legal/finanzas, analytics agent.*
* **Módulo B — Desarrollo del proyecto**
  *Planificación, herramientas, stack, integración CI/CD.*
* **Módulo C — Benchmarks**
  *Comparación BM25 vs híbrido vs dense-only; costes/token; latencia p95.*
* **Módulo D — Documentación profesional**
  *README, API docs, métricas, decisiones técnicas.*
* **Módulo E — Simulacros de entrevistas**
  *Preguntas nivel básico/intermedio/avanzado, live coding, diseño de sistema.*
* **Módulo F — Carrera y mercado**
  *Panorama global, roles mejor pagados, geografías, tips de negociación.*
* **Capstone final:** entrega de un proyecto completo con repo público/privado, documentación, benchmarks y video demo.

### 4.2 Funcionalidades del portal

* **Guía de capstones:** descripciones + requisitos + datasets de ejemplo.
* **Benchmarks scripts:** `make bench` para comparar retrieval y coste.
* **Plantillas de documentación:** `README.md`, `system-design.md`, `pitch-deck.pptx`.
* **Simulacros guiados:** bancos de preguntas + rúbrica de evaluación.
* **Recursos laborales:** plantillas de CV, ejemplos de portfolios, enlaces a bolsas de trabajo.

---

## 5. Requisitos no funcionales

### 5.1 UX

* Interfaz clara para elegir capstone.
* Progreso gamificado (completar hitos del proyecto).
* Checklists por módulo.

### 5.2 SEO

* Keywords: “capstones IA”, “benchmark RAG”, “AI Engineer portfolio”, “entrevistas IA dev”.
* Schema.org: `Course`, `EducationalOccupationalProgram`.

### 5.3 Performance

* Datasets optimizados (repos pequeños).
* Benchmarks que corren en < 10 min en laptop estándar.

---

## 6. Roadmap de desarrollo

| Semana | Entregable                                  |
| ------ | ------------------------------------------- |
| 1      | Landing + Guía de capstones                 |
| 2      | Módulo B (desarrollo)                       |
| 3      | Módulo C (benchmarks)                       |
| 4      | Módulo D (documentación)                    |
| 5      | Módulo E (simulacros entrevistas)           |
| 6      | Módulo F (mercado laboral) + Capstone final |

---

## 7. Recursos incluidos

* **Datasets de ejemplo** para distintos capstones (documentación técnica, legal, corporativa).
* **Scripts de benchmark** (`bench.py`, `eval-runner.json`).
* **Plantillas de doc y pitches**.
* **Banco de preguntas de entrevista** (básico, intermedio, avanzado).
* **Checklist de portfolio**.

---

## 8. Entregables del alumno

* Repo capstone completo (código, doc, CI/CD, seguridad, métricas).
* Benchmarks con gráficos comparativos.
* Documentación estilo profesional (`README`, `system-design.md`).
* Pitch técnico (video o presentación).
* Feedback de simulacros de entrevista.

---

## 9. Glosario (extracto)

* **Capstone:** proyecto final integrador que consolida todo lo aprendido.
* **Benchmark:** prueba comparativa con métricas estandarizadas.
* **Pitch técnico:** presentación breve del sistema y sus métricas ante evaluadores.
* **Portfolio:** conjunto de proyectos y materiales que demuestran habilidades.

---

## 10. Riesgos y mitigaciones

* **Riesgo:** alumnos eligen capstones demasiado ambiciosos.

  * **Mitigación:** opciones pre-diseñadas de dificultad escalonada.
* **Riesgo:** falta de experiencia en entrevistas.

  * **Mitigación:** simulacros guiados + feedback.
* **Riesgo:** dispersión en mercado laboral.

  * **Mitigación:** guías regionales + enfoque en roles concretos.

---

## 11. KPI de seguimiento interno

* % de alumnos que completan capstone.
* % que suben su portfolio a GitHub/LinkedIn.
* % que pasan simulacros internos.
* % que aplican a posiciones después de 3 meses.

---

## 12. Cierre

El **Portal 4** transforma al alumno en un profesional listo para el mercado, con un **capstone sólido, benchmarks y documentación profesional**, además de **confianza para entrevistas técnicas**.
Es el paso final para diferenciarse como **AI Engineer preparado para entornos reales y empresas globales**.

---
