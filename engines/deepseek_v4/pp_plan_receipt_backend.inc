void DeviceExpertCache::begin_pp_plan_receipt(uint64_t nonce,bool enabled) {
    if(!enabled&&!pp_receipt_available_)return;
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    const auto begin=reinterpret_cast<PpPlanReceiptBegin>(ggml_backend_reg_get_proc_address(reg,pp_plan_receipt_begin_symbol));
    if(!begin||!begin(backend_,nonce,enabled))throw std::runtime_error("PP receipt begin/retire requires a quiescent existing owner");
    if(enabled){pp_receipt_nonce_=nonce;pp_receipt_available_=true;}
    else if(nonce!=pp_receipt_nonce_)pp_receipt_available_=false;
}
const PpPlanReceiptPod* DeviceExpertCache::read_pp_plan_receipt() const {
    if(!pp_receipt_available_)return nullptr;
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    const auto read=reinterpret_cast<PpPlanReceiptRead>(ggml_backend_reg_get_proc_address(reg,pp_plan_receipt_read_symbol));
    const PpPlanReceiptPod* result=nullptr;
    if(!read||!read(backend_,&result))throw std::runtime_error("PP host plan receipt read unavailable");
    return result;
}
