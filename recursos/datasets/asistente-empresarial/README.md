# 🏢 Datasets - Asistente Empresarial

## 📋 Descripción

Datasets curados para desarrollar un asistente empresarial con capacidades de Q&A corporativo, onboarding inteligente, gestión de conocimiento y compliance.

## 📁 Datasets Incluidos

### 1. 📋 Políticas de Recursos Humanos
**Archivo:** `hr_policies.json`  
**Descripción:** 150+ políticas corporativas estructuradas  
**Uso:** Base de conocimiento para consultas de empleados

```json
{
  "policy_id": "HR-VAC-001",
  "title": "Política de Vacaciones y Tiempo Libre",
  "category": "benefits",
  "department": "human_resources",
  "effective_date": "2024-01-01",
  "last_updated": "2024-08-01",
  "version": "3.1",
  "content": "Los empleados acumulan tiempo de vacaciones basado en su antigüedad...",
  "key_points": [
    "Acumulación: 1.67 días por mes trabajado",
    "Máximo carry-over: 5 días al año siguiente",
    "Aprobación requerida con 2 semanas de anticipación",
    "Política de uso obligatorio: mínimo 5 días consecutivos por año"
  ],
  "eligibility": {
    "employment_type": ["full_time", "part_time"],
    "minimum_tenure": "90_days",
    "excluded_roles": ["contractor", "intern"]
  },
  "approval_workflow": [
    {"step": 1, "approver": "direct_manager", "required": true},
    {"step": 2, "approver": "hr_representative", "required_if": "duration > 10 days"}
  ],
  "related_policies": ["HR-SICK-001", "HR-LEAVE-001"],
  "access_level": "all_employees",
  "compliance_requirements": ["local_labor_law", "company_standards"]
}
```

### 2. 📖 Manual del Empleado
**Archivo:** `employee_handbook.json`  
**Descripción:** Secciones completas del manual corporativo  
**Uso:** Onboarding y consultas generales

```json
{
  "section_id": "EH-ONBOARD-001",
  "title": "Proceso de Onboarding para Nuevos Empleados",
  "category": "onboarding",
  "content": "Bienvenido a la empresa. Este proceso te guiará durante tus primeras semanas...",
  "subsections": [
    {
      "title": "Primer Día",
      "checklist": [
        "Recoger credenciales de acceso en recepción",
        "Completar formularios I-9 y W-4",
        "Sesión de bienvenida con HR (9:00 AM)",
        "Setup de estación de trabajo con IT",
        "Almuerzo de bienvenida con el equipo"
      ],
      "contacts": [
        {"role": "HR Buddy", "name": "María González", "extension": "1234"},
        {"role": "IT Support", "name": "Carlos Rodríguez", "extension": "5678"}
      ]
    },
    {
      "title": "Primera Semana",
      "learning_modules": [
        "Cultura y valores corporativos",
        "Políticas de seguridad y compliance",
        "Herramientas y sistemas empresariales",
        "Estructura organizacional"
      ],
      "meetings": [
        {"day": 2, "meeting": "1:1 con manager directo"},
        {"day": 3, "meeting": "Presentación del equipo"},
        {"day": 5, "meeting": "Revisión de progreso con HR"}
      ]
    }
  ],
  "role_specific_variations": {
    "engineering": {
      "additional_steps": [
        "Setup de ambiente de desarrollo",
        "Acceso a repositorios de código",
        "Sesión de arquitectura técnica"
      ]
    },
    "sales": {
      "additional_steps": [
        "Training en CRM",
        "Certificación de producto",
        "Shadowing de calls con clientes"
      ]
    }
  },
  "success_metrics": [
    "Completar todos los módulos de training en 2 semanas",
    "Pasar evaluación de compliance con 80%+",
    "Feedback positivo en revisión de 30 días"
  ]
}
```

### 3. 🔒 Documentos de Compliance
**Archivo:** `compliance_docs.json`  
**Descripción:** Procedimientos de compliance y regulaciones  
**Uso:** Verificación automática y alertas

```json
{
  "compliance_id": "COMP-GDPR-001",
  "regulation": "GDPR - General Data Protection Regulation",
  "jurisdiction": "European Union",
  "effective_date": "2018-05-25",
  "last_review": "2024-05-01",
  "risk_level": "high",
  "description": "Procedimientos para manejo de datos personales de ciudadanos EU",
  "requirements": [
    {
      "requirement_id": "GDPR-CONSENT",
      "title": "Consentimiento Explícito",
      "description": "Obtener consentimiento claro y específico antes de procesar datos personales",
      "implementation": [
        "Implementar checkboxes opt-in en formularios",
        "Mantener registros de consentimiento con timestamp",
        "Proporcionar mecanismo fácil para retirar consentimiento"
      ],
      "verification_method": "audit_consent_records",
      "responsible_team": "legal_and_privacy"
    },
    {
      "requirement_id": "GDPR-DATA-MINIMIZATION",
      "title": "Minimización de Datos",
      "description": "Recopilar solo datos necesarios para el propósito específico",
      "implementation": [
        "Revisar formularios de captura de datos",
        "Eliminar campos no esenciales",
        "Documentar justificación para cada campo"
      ],
      "verification_method": "data_mapping_audit",
      "responsible_team": "data_protection_officer"
    }
  ],
  "penalties": {
    "minor_violation": "2% of annual global turnover or €10M",
    "major_violation": "4% of annual global turnover or €20M"
  },
  "monitoring": {
    "frequency": "quarterly",
    "next_audit": "2024-11-01",
    "responsible_auditor": "external_privacy_consultant"
  },
  "related_policies": ["PRIV-001", "DATA-RET-001"],
  "training_requirements": {
    "all_employees": "annual_gdpr_awareness",
    "data_handlers": "quarterly_technical_training",
    "managers": "bi_annual_leadership_briefing"
  }
}
```

### 4. ❓ FAQs Corporativas
**Archivo:** `corporate_faqs.json`  
**Descripción:** Preguntas frecuentes con respuestas expertas  
**Uso:** Training del modelo y validación de respuestas

```json
{
  "faq_id": "FAQ-001",
  "category": "benefits",
  "question": "¿Cuántos días de vacaciones tengo si llevo 3 años en la empresa?",
  "answer": "Según la política de vacaciones vigente, los empleados con 3 años de antigüedad tienen derecho a 20 días hábiles de vacaciones base, más 3 días adicionales por antigüedad (después de 2 años), para un total de 23 días de vacaciones por año calendario.",
  "detailed_explanation": "El cálculo se basa en:\n- 20 días base para todos los empleados full-time\n- 1 día adicional por cada año completo después del segundo año\n- Máximo de 30 días totales (incluyendo días por antigüedad)\n- Los días se acumulan mensualmente (1.92 días/mes para 23 días anuales)",
  "related_questions": [
    "¿Puedo llevar días de vacaciones al año siguiente?",
    "¿Cómo solicito mis vacaciones?",
    "¿Qué pasa si no uso todos mis días de vacaciones?"
  ],
  "source_documents": ["HR-VAC-001", "Employee_Handbook_Section_4.2"],
  "last_updated": "2024-08-01",
  "confidence_score": 0.95,
  "access_level": "all_employees",
  "variations": [
    {
      "question_variant": "vacation days after 3 years",
      "answer_summary": "23 vacation days total (20 base + 3 seniority)"
    }
  ]
}
```

### 5. 🏗️ Procedimientos Operacionales
**Archivo:** `operational_procedures.json`  
**Descripción:** SOPs y procedimientos de trabajo  
**Uso:** Automatización de workflows y guías de proceso

```json
{
  "procedure_id": "SOP-IT-001",
  "title": "Procedimiento de Solicitud de Acceso a Sistemas",
  "department": "information_technology",
  "category": "access_management",
  "classification": "internal",
  "version": "2.3",
  "effective_date": "2024-06-01",
  "review_cycle": "annual",
  "next_review": "2025-06-01",
  "purpose": "Establecer proceso estándar para solicitar y aprobar acceso a sistemas corporativos",
  "scope": "Aplica a todos los empleados, contratistas y terceros que requieren acceso a sistemas",
  "procedure_steps": [
    {
      "step_number": 1,
      "title": "Solicitud Inicial",
      "description": "Empleado o manager inicia solicitud en portal de ServiceNow",
      "required_information": [
        "Nombre completo del solicitante",
        "Departamento y rol",
        "Sistemas específicos requeridos",
        "Justificación de negocio",
        "Fecha de inicio requerida"
      ],
      "estimated_duration": "5 minutos",
      "responsible_party": "solicitante_o_manager"
    },
    {
      "step_number": 2,
      "title": "Aprobación de Manager",
      "description": "Manager directo revisa y aprueba la solicitud",
      "approval_criteria": [
        "Necesidad legítima de negocio",
        "Principio de menor privilegio",
        "Compliance con políticas de seguridad"
      ],
      "sla": "2 business days",
      "escalation": "director_level_after_3_days"
    },
    {
      "step_number": 3,
      "title": "Revisión de Seguridad",
      "description": "Equipo de seguridad evalúa riesgos y compliance",
      "security_checks": [
        "Verificación de background check",
        "Evaluación de nivel de acceso apropiado",
        "Revisión de segregación de funciones"
      ],
      "sla": "3 business days",
      "bypass_conditions": ["emergency_access_with_director_approval"]
    }
  ],
  "success_criteria": [
    "100% de solicitudes procesadas dentro de SLA",
    "Zero violaciones de compliance en auditorías",
    "95%+ satisfaction score en encuestas de usuario"
  ],
  "related_documents": ["SEC-POL-001", "IT-STD-002"],
  "tools_required": ["ServiceNow", "Active Directory", "Privileged Access Management"]
}
```

## 🎯 Casos de Uso por Dataset

### Q&A Corporativo
- **HR Policies:** Consultas sobre beneficios y políticas
- **Employee Handbook:** Información de onboarding y procedimientos
- **Corporate FAQs:** Validación de respuestas y training

### Compliance y Auditoría
- **Compliance Docs:** Verificación automática de regulaciones
- **Operational Procedures:** Workflows automatizados
- **HR Policies:** Control de acceso por roles

### Onboarding Inteligente
- **Employee Handbook:** Guías personalizadas por rol
- **Operational Procedures:** Checklists de tareas
- **Corporate FAQs:** Respuestas a preguntas comunes

## 📊 Métricas de Evaluación

### Precisión de Respuestas
```python
# Métrica: Accuracy vs. expert answers
def evaluate_qa_accuracy(predicted_answer, expert_answer, source_docs):
    return {
        'semantic_similarity': cosine_similarity(predicted_answer, expert_answer),
        'factual_accuracy': fact_check_against_sources(predicted_answer, source_docs),
        'completeness': coverage_score(predicted_answer, expert_answer),
        'citation_accuracy': validate_citations(predicted_answer, source_docs)
    }
```

### Control de Acceso
```python
# Métrica: RBAC compliance
def evaluate_access_control(user_role, requested_document, access_granted):
    return {
        'rbac_compliance': check_role_permissions(user_role, requested_document),
        'over_privilege': detect_excessive_access(user_role, access_granted),
        'audit_trail_completeness': validate_access_logging(user_role, requested_document)
    }
```

## 🚀 Uso Recomendado

### Semana 1: Q&A Básico
- Implementar búsqueda en HR Policies (20 políticas)
- Sistema básico de citaciones
- Interface de chat simple

### Semana 2: Onboarding Inteligente
- Integrar Employee Handbook
- Personalización por rol de usuario
- Checklists automáticos

### Semana 3: Compliance y Seguridad
- Agregar Compliance Docs
- Implementar RBAC
- Audit trail completo

### Semana 4: Optimización
- Integrar todos los datasets
- Fine-tuning para precisión
- Dashboard de métricas empresariales

---

**📥 Descarga:** Los datasets están disponibles en formato JSON con estructura consistente para fácil integración.
