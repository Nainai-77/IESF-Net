"""Rule lexicons of the institutional-action candidate extractor (Section 3.6.2, Table 4).

The extractor identifies category anchors and operative expressions within the
same clause.  Mechanism directions refer to changes in the institutional
arrangement: an increase in Allocation denotes a larger permit supply, an
increase in Compliance a stricter obligation, etc. (Section 3.2.2).
"""
from __future__ import annotations

import re

CATEGORIES = ["Allocation", "Compliance", "Offsets", "Coverage", "Trading", "MRV", "Finance"]

CATEGORY_ANCHORS: dict[str, list[str]] = {
    # cap, benchmark, reserve and allocation adjustment
    "Allocation": ["配额分配", "分配方案", "配额总量", "总量设定", "总量控制", "免费分配", "有偿分配", "配额拍卖",
                   "拍卖配额", "基准线", "基准值", "历史强度", "历史排放", "配额预留", "储备配额", "政府预留",
                   "配额调整", "初始配额", "核定配额", "配额核定", "配额发放", "配额投放", "预分配", "配额回购"],
    # surrender deadline, obligation ratio and penalty
    "Compliance": ["履约", "清缴", "配额清缴", "上缴配额", "履约期", "履约截止", "履约周期", "清缴比例",
                   "未足额清缴", "未按时履约", "差额部分", "责令履行", "罚款", "处罚"],
    # eligibility, offset ratio, restart and methodology
    "Offsets": ["CCER", "国家核证自愿减排量", "核证自愿减排量", "抵消", "抵销", "自愿减排", "方法学",
                "碳普惠", "PHCER", "林业碳汇", "减排项目"],
    # new sectors and participant scope
    "Coverage": ["纳入", "纳管", "控排企业", "重点排放单位", "管控单位", "覆盖范围", "覆盖行业", "纳入标准",
                 "纳入门槛", "扩大行业", "扩容", "新增行业", "准入", "个人投资者", "机构投资者", "市场参与人"],
    # auction, price limits and market-making arrangements
    "Trading": ["交易规则", "交易细则", "涨跌幅", "价格限制", "竞价", "挂牌", "协议转让", "定价转让", "大宗交易",
                "做市", "交易时间", "交易方式", "风险控制", "价格平抑", "拍卖", "交易品种", "交易制度", "交易管理"],
    # monitoring, reporting, verification and enforcement
    "MRV": ["监测计划", "排放监测", "排放报告", "温室气体排放报告", "核查", "第三方核查", "核查机构", "数据质量",
            "执法", "监督检查", "违规交易", "查处", "现场检查", "信息披露"],
    # pledging, financing and related trading rules
    "Finance": ["质押", "回购", "托管", "碳金融", "融资", "碳基金", "碳债券", "碳远期", "碳期货", "碳资产",
                "借碳", "授信", "碳保险", "碳信贷", "绿色金融"],
}

OPERATIVE = ["印发", "发布", "施行", "实施", "执行", "生效", "调整", "修订", "修改", "废止", "取消", "暂停",
             "恢复", "重启", "启动", "延期", "推迟", "延长", "提前", "提高", "降低", "增加", "减少", "扩大", "缩小",
             "收紧", "放宽", "上调", "下调", "应当", "不得", "须", "必须", "要求", "决定", "规定", "明确", "开展",
             "实行", "建立"]

NORMATIVE = re.compile(r"(第[一二三四五六七八九十百零〇\d]+条|应当|不得|必须|须于|方可)")
FORMAL_INSTRUMENT = re.compile(r"《([^》]{2,80}?(办法|规定|通知|方案|细则|规则|意见|公告|决定|条例|指南|指引|规划))[^》]{0,12}》")
TITLE_INSTRUMENT = re.compile(r"(办法|规定|通知|方案|细则|规则|意见|公告|决定|条例|指南|指引)")
AMEND_CANCEL = ["修订", "修改", "废止", "取消", "撤销", "失效", "停止执行", "变更"]
CLAUSE_SPLIT = re.compile(r"[。；;！？!?\n]+")

GENERIC_MECHANISM = {
    "reschedule": ["延期", "推迟", "延长", "提前", "顺延", "截止日期调整", "时间调整", "调整至", "延至"],
    "increase": ["提高", "增加", "扩大", "上调", "新增", "加大", "提升", "追加", "增发"],
    "decrease": ["降低", "减少", "缩小", "下调", "削减", "缩减", "压减", "收窄"],
}

CATEGORY_MECHANISM = {
    "Allocation": {"increase": ["增发", "追加配额", "投放", "放宽分配", "提高免费"],
                   "decrease": ["收紧", "从严", "注销", "回收配额", "降低免费", "总量下降"]},
    "Compliance": {"increase": ["提高清缴比例", "加大处罚", "从严", "缩短履约", "强化履约"],
                   "decrease": ["豁免", "减免", "放宽", "免于处罚", "降低清缴"]},
    "Offsets": {"increase": ["提高抵消比例", "重启", "恢复", "放宽", "扩大抵消"],
                "decrease": ["限制", "降低抵消比例", "暂停", "收紧", "不得用于抵消"]},
    "Coverage": {"increase": ["纳入", "扩大覆盖", "新增行业", "降低纳入门槛", "扩容"],
                 "decrease": ["退出", "剔除", "移出", "不再纳入", "提高纳入门槛"]},
    "Trading": {"increase": ["放宽涨跌幅", "延长交易时间", "引入做市", "新增交易品种", "扩大涨跌幅"],
                "decrease": ["收窄涨跌幅", "暂停交易", "限制交易", "熔断", "停牌"]},
    "MRV": {"increase": ["加强", "强化", "从严", "严厉", "加大检查"],
            "decrease": ["简化", "放宽", "减免核查", "免于核查"]},
    "Finance": {"increase": ["推出", "支持", "创新", "扩大", "试点开展"],
                "decrease": ["限制", "暂停", "收紧", "叫停"]},
}

NATIONAL = ["全国碳市场", "全国碳排放权", "全国统一", "生态环境部", "国务院", "国家发展改革委", "国家发改委", "国家主管部门"]
CROSS_MARKET = ["跨区域", "跨市场", "各试点", "试点地区", "试点省市", "地方碳市场", "衔接", "互认", "互通"]
REGION_NAMES = {
    "Hubei": ["湖北", "武汉"], "Guangdong": ["广东", "广州"], "Shanghai": ["上海"], "Shenzhen": ["深圳"],
    "Beijing": ["北京"], "Tianjin": ["天津"], "Chongqing": ["重庆"], "Fujian": ["福建"],
}

REGULATOR = ["生态环境部", "生态环境厅", "生态环境局", "发展和改革委员会", "发改委", "发展改革委", "人民政府",
             "人大常委会", "人民代表大会", "国务院", "主管部门", "办公厅", "应对气候变化"]
EXCHANGE = ["交易所", "交易中心", "排放权交易", "环境能源交易所", "联合产权交易所", "碳排放权登记结算"]
ISSUER_PATTERN = re.compile(r"([一-龥]{2,20}(生态环境部|生态环境厅|生态环境局|发展和改革委员会|发展改革委|发改委|"
                            r"人民政府|人大常委会|交易所|交易中心|办公厅|登记结算[一-龥]{0,6}))")

EFFECTIVE_PATTERNS = [
    re.compile(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起(?:正式)?(?:施行|实施|执行|生效)"),
    re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日起(?:正式)?(?:施行|实施|执行|生效)"),
    re.compile(r"于(\d{4})年(\d{1,2})月(\d{1,2})日(?:正式)?(?:施行|实施|生效|开始)"),
    re.compile(r"(?:截止|截至|于)(\d{4})年(\d{1,2})月(\d{1,2})日(?:前|之前)"),
    re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})起(?:施行|实施|执行|生效)"),
]
EFFECTIVE_ON_RELEASE = re.compile(r"自(?:发布|印发|公布)之日起(?:施行|实施|执行|生效)")
