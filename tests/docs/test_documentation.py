"""Tests para validar la calidad de la documentación."""
import re
from pathlib import Path
from typing import List, Set
import pytest

class TestDocumentationStructure:
    """Tests para estructura de documentación."""
    
    def test_all_modules_have_readme(self):
        """Verifica que todos los módulos tengan README."""
        modulos_path = Path("modulos")
        if not modulos_path.exists():
            pytest.skip("Directorio modulos no existe")
            
        module_dirs = [d for d in modulos_path.iterdir() 
                      if d.is_dir() and d.name.startswith("modulo-")]
        
        assert len(module_dirs) >= 6, "Debe haber al menos 6 módulos (A-F)"
        
        for module_dir in module_dirs:
            readme_path = module_dir / "README.md"
            assert readme_path.exists(), f"Falta README en {module_dir.name}"
            
            # README no debe estar vacío
            content = readme_path.read_text(encoding="utf-8")
            assert len(content.strip()) > 100, f"README muy corto en {module_dir.name}"
            
    def test_consistent_module_structure(self):
        """Verifica estructura consistente entre módulos."""
        modulos_path = Path("modulos")
        if not modulos_path.exists():
            pytest.skip("Directorio modulos no existe")
            
        expected_sections = ["Contenido del Módulo", "Actividades", "Recursos"]
        
        for module_dir in modulos_path.iterdir():
            if module_dir.is_dir() and module_dir.name.startswith("modulo-"):
                readme_path = module_dir / "README.md"
                content = readme_path.read_text(encoding="utf-8")
                
                for section in expected_sections:
                    assert section in content, f"Falta sección '{section}' en {module_dir.name}"
                    
    def test_navigation_consistency(self):
        """Verifica que la navegación sea consistente."""
        main_readme = Path("README.md")
        modules_readme = Path("modulos/README.md")
        
        assert main_readme.exists()
        
        if not modules_readme.exists():
            pytest.skip("README de módulos no existe")
        
        # Verificar links a módulos en README principal
        main_content = main_readme.read_text(encoding="utf-8")
        module_links = re.findall(r'\[.*?\]\((\.\/modulos\/modulo-[a-f]-.*?\/)\)', main_content)
        
        assert len(module_links) >= 6, "Debe haber links a todos los módulos A-F"

class TestLinkValidation:
    """Tests para validación de enlaces."""
    
    def get_all_markdown_files(self) -> List[Path]:
        """Obtiene todos los archivos markdown del proyecto."""
        markdown_files = []
        for pattern in ["**/*.md"]:
            markdown_files.extend(Path(".").glob(pattern))
        return markdown_files
        
    def extract_links(self, content: str) -> List[str]:
        """Extrae todos los enlaces de un contenido markdown."""
        # Enlaces formato [texto](url)
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        # Enlaces formato <url>
        links.extend(re.findall(r'<(https?://[^>]+)>', content))
        return links
        
    def test_internal_links_validity(self):
        """Verifica que todos los enlaces internos sean válidos."""
        broken_links = []
        
        for md_file in self.get_all_markdown_files():
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
                
            links = self.extract_links(content)
            
            for link in links:
                if link.startswith(("./", "../", "/")):
                    # Link relativo o absoluto interno
                    if link.startswith("/"):
                        target_path = Path("." + link)
                    else:
                        target_path = (md_file.parent / link).resolve()
                        
                    if not target_path.exists():
                        broken_links.append(f"{md_file}: {link}")
                        
        assert len(broken_links) == 0, f"Enlaces rotos encontrados:\n" + "\n".join(broken_links)
        
    @pytest.mark.network
    def test_external_links_accessibility(self):
        """Verifica que enlaces externos sean accesibles."""
        try:
            import requests
        except ImportError:
            pytest.skip("requests no disponible")
            
        external_links = set()
        
        for md_file in self.get_all_markdown_files():
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
                
            links = self.extract_links(content)
            
            for link in links:
                if link.startswith(("http://", "https://")):
                    external_links.add(link)
                    
        # Limitar a 10 enlaces para no sobrecargar
        test_links = list(external_links)[:10]
        broken_external = []
        
        for link in test_links:
            try:
                response = requests.head(link, timeout=10, allow_redirects=True)
                if response.status_code >= 400:
                    broken_external.append(f"{link}: HTTP {response.status_code}")
            except requests.RequestException as e:
                broken_external.append(f"{link}: {str(e)}")
                
        if broken_external:
            print(f"Enlaces externos con problemas:\n" + "\n".join(broken_external))
            # No fallar el test por enlaces externos, solo reportar

class TestCodeBlocks:
    """Tests para validación de bloques de código."""
    
    def test_code_blocks_have_language(self):
        """Verifica que todos los code blocks tengan lenguaje especificado."""
        issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            
            # Encontrar bloques de código
            code_blocks = re.finditer(r'```(\w*)\n', content)
            
            for i, match in enumerate(code_blocks):
                language = match.group(1)
                if not language.strip():
                    line_number = content[:match.start()].count('\n') + 1
                    issues.append(f"{md_file}:{line_number} - Code block sin lenguaje")
                    
        assert len(issues) == 0, f"Code blocks sin lenguaje:\n" + "\n".join(issues)
        
    def test_python_code_syntax(self):
        """Verifica que el código Python tenga sintaxis válida."""
        syntax_errors = []
        
        for md_file in Path(".").glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            
            # Extraer bloques de código Python
            python_blocks = re.finditer(r'```python\n(.*?)```', content, re.DOTALL)
            
            for i, match in enumerate(python_blocks):
                code = match.group(1)
                
                # Skip code blocks que son solo imports o comentarios
                if len(code.strip()) < 10:
                    continue
                    
                try:
                    compile(code, f"{md_file}:block_{i}", "exec")
                except SyntaxError as e:
                    line_start = content[:match.start()].count('\n') + 1
                    syntax_errors.append(
                        f"{md_file}:{line_start} - Syntax error: {e.msg}"
                    )
                    
        assert len(syntax_errors) == 0, f"Errores de sintaxis Python:\n" + "\n".join(syntax_errors)

class TestContentQuality:
    """Tests para calidad de contenido."""
    
    def test_heading_hierarchy(self):
        """Verifica jerarquía correcta de headings."""
        hierarchy_issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            
            # Extraer headings con sus niveles
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            
            prev_level = 0
            for heading_match in headings:
                current_level = len(heading_match[0])
                
                # El primer heading debe ser H1
                if prev_level == 0 and current_level != 1:
                    hierarchy_issues.append(f"{md_file}: Primer heading debe ser H1")
                    
                # No saltar más de un nivel
                elif current_level > prev_level + 1:
                    hierarchy_issues.append(
                        f"{md_file}: Salto de H{prev_level} a H{current_level}"
                    )
                    
                prev_level = current_level
                
        assert len(hierarchy_issues) == 0, f"Problemas de jerarquía:\n" + "\n".join(hierarchy_issues)
        
    def test_minimum_content_length(self):
        """Verifica que los archivos tengan contenido mínimo."""
        short_files = []
        min_words = 100
        
        for md_file in Path(".").glob("**/*.md"):
            if md_file.name.startswith("test_"):
                continue
                
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            
            # Contar palabras excluyendo código y metadatos
            text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            text_content = re.sub(r'`[^`]+`', '', text_content)
            
            words = len(text_content.split())
            
            if words < min_words:
                short_files.append(f"{md_file}: {words} palabras (mínimo {min_words})")
                
        assert len(short_files) == 0, f"Archivos muy cortos:\n" + "\n".join(short_files)
        
    def test_consistent_spanish_language(self):
        """Verifica consistencia en el uso del español."""
        english_patterns = [
            r'\bthe\b', r'\band\b', r'\bor\b', r'\bbut\b', r'\bfor\b',
            r'\bwith\b', r'\bfrom\b', r'\bthis\b', r'\bthat\b', r'\bwhen\b'
        ]
        
        issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            
            # Excluir bloques de código
            text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            text_content = re.sub(r'`[^`]+`', '', text_content)
            
            for pattern in english_patterns:
                matches = re.finditer(pattern, text_content, re.IGNORECASE)
                for match in matches:
                    line_num = text_content[:match.start()].count('\n') + 1
                    issues.append(f"{md_file}:{line_num} - Posible texto en inglés: '{match.group()}'")
                    
        # Permitir algunas ocurrencias pero reportar si hay muchas
        if len(issues) > 20:
            print(f"Advertencia: {len(issues)} posibles textos en inglés detectados")
            print("Primeros 10:")
            for issue in issues[:10]:
                print(f"  {issue}")
