-- EPUB 전용: \label{eq:...} / \eqref{eq:...} 를 수식 번호로 바꾼다 (#210).
--
-- pandoc 의 LaTeX → EPUB3 변환은 equation 환경에 번호를 매기지 않아서
-- \eqref{eq:ch05-01} 이 "[eq:ch05-01]" 라벨 그대로 노출되고, 링크 대상 id 도 없다.
-- PDF(book 클래스)와 같은 "(장.순번)" 번호를 직접 매겨
--   1) 수식 뒤에 번호를 붙이고 id 를 달고,
--   2) \eqref 링크 글자를 그 번호로 바꾼다.
-- texmath 는 \tag{} 를 버리므로 번호는 수식 밖 텍스트로 붙인다.

local numbers = {}

local function label_of(text)
  return text:match("\\label{([^}]*)}")
end

-- 1차: 문서 순서대로 장 번호와 수식 순번을 센다.
-- 번호 있는 \chapter 만 장으로 세고, \chapter*(Phase 간지 등)는 건너뛴다.
local function collect(doc)
  local chapter, count = 0, 0
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then
      if not block.classes:includes("unnumbered") then
        chapter = chapter + 1
        count = 0
      end
    else
      pandoc.walk_block(block, {
        Math = function(el)
          local label = el.mathtype == "DisplayMath" and label_of(el.text)
          if label then
            count = count + 1
            numbers[label] = string.format("%d.%d", chapter, count)
          end
        end,
      })
    end
  end
end

-- 2차: 수식에 id·번호를 달고, \eqref 링크 글자를 번호로 바꾼다.
local function render(doc)
  return doc:walk({
    Math = function(el)
      local label = el.mathtype == "DisplayMath" and label_of(el.text)
      if not label or not numbers[label] then
        return nil
      end
      el.text = el.text:gsub("%s*\\label{[^}]*}%s*", "")
      return pandoc.Span(
        { el, pandoc.Span({ pandoc.Str("(" .. numbers[label] .. ")") }, pandoc.Attr("", { "eqno" })) },
        pandoc.Attr(label, { "equation" })
      )
    end,
    Link = function(el)
      if el.attributes["reference-type"] ~= "eqref" then
        return nil
      end
      local number = numbers[el.attributes["reference"]]
      if not number then
        io.stderr:write("eqref-numbers: 정의되지 않은 수식 참조 " .. el.attributes["reference"] .. "\n")
        return nil
      end
      el.content = { pandoc.Str("(" .. number .. ")") }
      return el
    end,
  })
end

function Pandoc(doc)
  collect(doc)
  return render(doc)
end
